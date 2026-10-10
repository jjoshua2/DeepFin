"""Pipelined game RPC consumer for the shared teacher dispatcher.

Resource/session admission stays with the existing owner. Its immutable raw
history loader and atomic companion publisher are supplied directly; no second
model loader, root encoder, watchdog, or publication schema is introduced.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
import hashlib
import argparse
import importlib.util
import json
import math
import os
import select
import time
from pathlib import Path
import sys
from typing import Any, TextIO

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_anti_engine.teacher_dispatch import TeacherDispatcher
from scripts.shared_teacher_generation import CompletedGameLabels
from scripts import ceres_raw_backend

RETAINED_SHA = {
    "retained": "ea1cb8da49f0b8f44a2000aae4276a5abaf922f4b25d866e6cd05628638027a3",
    "comparison": "f2462476262a9cc2b06a00bca3a64f989075ea85335326c4b131d6ec0689ca2a",
    "publisher": "4e34271ddacdb13a5e3a13f55eee24479ffa3a9bc049763ef8f16e789374ff47",
}


def bind_retained_ceres_history(
    *, retained: Any, comparison: Any, publisher: Any, api: Any,
    units: dict[str, Path], model: dict[str, Any], fake_cpu: bool,
    physical_batch: int, provider: dict[str, Any] | Callable[[], dict[str, Any] | None] | None,
    authorized_raw_roots: dict[str, tuple[Path, ...]] | None = None,
) -> tuple[Callable[[dict[str, Any]], Sequence[Any]],
           Callable[[dict[str, Any], tuple[Any, ...]], dict[str, Any]]]:
    """Reuse pinned full-history loader, raw packer and create-only publisher.

    The session and provider proof remain caller-owned. Mixed-game physical
    accounting belongs to the shared service, never to individual game receipts.
    Old receipts without the explicit unit namespace are rejected unchanged.
    """
    for module, key in ((retained, "retained"), (comparison, "comparison"), (publisher, "publisher")):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != RETAINED_SHA[key]:
            raise ValueError("retained Ceres source binding changed")
    units = dict(units)
    if (not 1 <= len(units) <= 4096 or physical_batch not in (32, 256, 512)
            or model != comparison.MODELS["ceres"]
            or len(set(units.values())) != len(units)
            or any(path.resolve() != path or not path.is_absolute() for path in units.values())):
        raise ValueError("explicit finite units and physical shape required")
    roster = tuple(units)
    allowed_roots = ({name: (path,) for name, path in units.items()} if authorized_raw_roots is None
             else {name: tuple(paths) for name, paths in authorized_raw_roots.items()})
    flat = [path for paths in allowed_roots.values() for path in paths]
    if (set(allowed_roots) != set(units) or len(set(flat)) != len(flat)
            or any(units[name] not in paths or not 1 <= len(paths) <= 8
                   or any(path.resolve() != path or not path.is_absolute() for path in paths)
                   for name, paths in allowed_roots.items())):
        raise ValueError("distinct finite authorized Ceres unit attempt roots required")
    expected: dict[int, tuple[dict[str, Any], Path, Path]] = {}
    np_api = api[0]

    def load(message: dict[str, Any]) -> Sequence[Any]:
        unit, game, request = message["unit_id"], message["game_id"], message["request_id"]
        raw = Path(message["raw_path"])
        if (unit not in units or type(game) is not int or not 0 <= game < 128
                or type(request) is not int or request != roster.index(unit) * 128 + game
                or request in expected
                or raw not in tuple(path / "games" / f"game_{game:08d}.npz" for path in allowed_roots[unit])):
            raise ValueError("exact per-unit local raw path required")
        roots, ids, feeds = retained.game_roots(comparison, api, raw, game, message["raw_sha256"])
        directory = raw.parent.parent / "ceres_companions"
        directory.mkdir(mode=0o700, exist_ok=True)
        if directory.resolve() != directory or directory.is_symlink():
            raise ValueError("canonical separate companion directory required")
        identity = {"schema": "BT4_GENERATED_ROOT_C3_COMPANION_V1", "unit_id": unit,
                        "raw_path": str(raw), "raw_sha256": message["raw_sha256"], "game_id": game,
                        "root_ids": ids, "feed_sha256": feeds, "real_rows": len(roots), "model": model, "fake_cpu": fake_cpu}
        receipt_path, labels = directory / (raw.stem + ".json"), directory / raw.name
        expected[request] = identity, receipt_path, labels
        if receipt_path.exists():
            retained.verify_companion(np_api, json.loads(receipt_path.read_text()), identity)
            return ()  # durable identity recovered without reinference
        if labels.exists():
            with np_api.load(labels, allow_pickle=False) as archive:
                orphan = json.loads(archive["companion_metadata"].tobytes().decode())
            orphan["labels"] = {"path": str(labels), "sha256": retained.sha(labels)}
            retained.verify_companion(np_api, orphan, identity)
            return ()
        return tuple(replace(root, slot_id=request * 1024 + index) for index, root in enumerate(roots))

    def publish(message: dict[str, Any], values: tuple[Any, ...]) -> dict[str, Any]:
        identity, receipt_path, labels = expected[message["request_id"]]
        if any(message[key] != identity[key] for key in ("unit_id", "game_id", "raw_path", "raw_sha256")):
            raise ValueError("publication message changed retained game identity")
        if receipt_path.exists():
            receipt = retained.verify_companion(np_api, json.loads(receipt_path.read_text()), identity)
        elif labels.exists():
            with np_api.load(labels, allow_pickle=False) as archive:
                receipt = json.loads(archive["companion_metadata"].tobytes().decode())
            receipt["labels"] = {"path": str(labels), "sha256": retained.sha(labels)}
            retained.verify_companion(np_api, receipt, identity)
            publisher.durable_json(receipt_path, receipt)
        else:
            if len(values) != identity["real_rows"]:
                raise ValueError("complete local game labels required before publication")
            if any(value.slot_id != message["request_id"] * 1024 + index for index, value in enumerate(values)):
                raise ValueError("publication changed exact global root routing")
            local = tuple(replace(value, slot_id=message["game_id"] * 1024 + index)
                          for index, value in enumerate(values))
            arrays = comparison.pack("ceres", local, np_api) if local else {
                "policy_logits": np_api.empty((0, 1858), np_api.float16),
                "value_logits": np_api.empty((0, 3), np_api.float16),
                "value2_logits": np_api.empty((0, 3), np_api.float16),
                "feed_sha256": np_api.array([], dtype="<U64"),
            }
            arrays["root_id"] = np_api.array(identity["root_ids"], dtype=np_api.int64).reshape((-1, 2))
            contract = comparison.label_contract("ceres", local) if local else None
            if contract is not None:
                contract = {**contract, "input_shape": [physical_batch, 64, 137]}
            receipt = {**identity, "provider": provider() if callable(provider) else provider, "label_contract": contract,
                       "physical_accounting_scope": "shared_service_successful_complete_calls",
                       "physical_rows": None, "padding_rows": None, "session_calls": None,
                       "admission_credit": 0}
            arrays["companion_metadata"] = np_api.frombuffer(json.dumps(receipt, sort_keys=True).encode(),
                                                              dtype=np_api.uint8).copy()
            publisher.durable_npz(np_api, labels, arrays)
            with np_api.load(labels, allow_pickle=False) as archive:
                if set(archive.files) != set(arrays) or any(not np_api.array_equal(archive[key], value)
                                                          for key, value in arrays.items()):
                    raise ValueError("durable raw Ceres exact readback changed")
            receipt["labels"] = {"path": str(labels), "sha256": retained.sha(labels)}
            publisher.durable_json(receipt_path, receipt)
            retained.verify_companion(np_api, receipt, identity)
        del expected[message["request_id"]]
        return {"companion": {"path": str(receipt_path), "sha256": retained.sha(receipt_path)},
                    "real_rows": identity["real_rows"]}

    return load, publish


def bind_ceres_raw_backend(
    *, session: Any, physical_batch: int = 32,
    gather_context: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    gather_indices: Callable[[np.ndarray, np.ndarray], np.ndarray],
    accounting: dict[str, int],
) -> Callable[[Sequence[Any]], Sequence[Any]]:
    """Bind the reviewed raw adapter to RPC without opening a teacher session.

    Default32 remains fixed. Explicit256/512 are isolated shape experiments;
    owner admission, pinned session/provider proof and GPU qualification precede
    production use. Accounting has four fixed keys for successful validated
    inference only; failed attempted calls are excluded, so it is not a budget
    meter. No call history is retained.
    """
    if type(physical_batch) is not int or physical_batch not in (32, 256, 512):
        raise ValueError("isolated physical batch must be32/256/512")
    if accounting:
        raise ValueError("fresh physical accounting required")
    accounting.update(calls=0, real_rows=0, padding_rows=0, physical_rows=0)

    def infer(roots: Sequence[Any]) -> Sequence[Any]:
        values, receipt = ceres_raw_backend.infer_raw_roots(
            roots, session=session, physical_batch=physical_batch,
            gather_context=gather_context, gather_indices=gather_indices,
        )
        for key in ("calls", "real_rows", "padding_rows", "physical_rows"):
            accounting[key] += getattr(receipt, key)
        return values

    return infer


def checked_ceres_backend(
    infer: Callable[[Sequence[Any]], Sequence[Any]],
) -> Callable[[Sequence[Any]], Sequence[Any]]:
    """Keep the frozen C3 raw three-head contract and validate exact routing."""
    def evaluate(roots: Sequence[Any]) -> Sequence[Any]:
        identities = [(root.slot_id, root.fen, np.ascontiguousarray(root.feed).tobytes()) for root in roots]
        values = tuple(infer(roots))
        if len(values) != len(roots):
            raise ValueError("Ceres exact root coverage changed")
        for root, value, (root_id, fen, feed_bytes) in zip(roots, values, identities):
            digest = hashlib.sha256(feed_bytes).hexdigest()
            if (root.feed.dtype != np.uint8 or root.feed.shape != (64, 137)
                    or root.slot_id != root_id or root.fen != fen
                    or np.ascontiguousarray(root.feed).tobytes() != feed_bytes
                    or value.slot_id != root_id or value.fen != fen
                    or value.feed.dtype != np.uint8 or value.feed.shape != (64, 137)
                    or value.feed_sha256 != digest or not np.array_equal(value.feed, root.feed)):
                raise ValueError("Ceres stale/misrouted root or feed identity")
            for name, width in (("policy_logits", 1858), ("value_logits", 3), ("value2_logits", 3)):
                array = getattr(value, name)
                if array.dtype != np.float16 or array.shape != (width,) or not np.isfinite(array).all():
                    raise ValueError("Ceres raw FP16 head contract changed: " + name)
        return values
    return evaluate


def serve_ceres_stream(
    incoming: TextIO, outgoing: TextIO, *, total_games: int, max_games: int,
    target_rows: int, max_rows: int, batch_wait_ms: float,
    poll_seconds: float, deadline_seconds: float,
    load_roots: Callable[[dict[str, Any]], Sequence[Any]],
    infer: Callable[[Sequence[Any]], Sequence[Any]],
    publish: Callable[[dict[str, Any], tuple[Any, ...]], dict[str, Any]],
    control: Callable[[], None] = lambda: None,
    unit_roster: tuple[str, ...] | None = None,
    fake_cpu: bool | None = None,
) -> dict[str, Any]:
    """Actual bounded JSON-line RPC: pipeline games, route complete durable replies.

    The independent fill deadline always drains low volume. select's existing
    response-poll cadence remains separate from that deadline. A stop/EOF stops
    admission and drains accepted requests; the owner-supplied guard runs each
    controller poll and before backend/publication. One held input request plus
    max_games admitted games bounds decoded histories and result storage.
    """
    if (unit_roster is not None and (not 1 <= len(unit_roster) <= 4096
            or len(set(unit_roster)) != len(unit_roster)
            or any(type(unit) is not str or not unit or len(unit.encode()) > 128 for unit in unit_roster)
            or total_games != len(unit_roster) * 128)):
        raise ValueError("exact finite service unit roster required")
    if (type(total_games) is not int or not 1 <= total_games <= 128 * (len(unit_roster) if unit_roster else 1)
            or not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 1
            or not math.isfinite(deadline_seconds) or deadline_seconds <= 0):
        raise ValueError("finite explicit RPC poll and operation budget required")
    messages: dict[int, dict[str, Any]] = {}
    responses: dict[int, dict[str, Any]] = {}
    evaluate = checked_ceres_backend(infer)

    def guarded_infer(roots: Sequence[Any]) -> Sequence[Any]:
        control()
        return evaluate(roots)

    def durable(game_id: int, raw_sha: str, values: tuple[Any, ...]) -> None:
        control()
        message = messages[game_id]
        if message["raw_sha256"] != raw_sha:
            raise ValueError("Ceres raw-game provenance changed")
        responses[game_id] = publish(message, values)

    dispatcher = TeacherDispatcher(guarded_infer, target_rows=target_rows, max_rows=max_rows, batch_wait_ms=batch_wait_ms)
    try:
        labels = CompletedGameLabels(dispatcher, max_games=max_games, total_games=total_games,
                                    max_game_rows=400, publish=durable)
    except BaseException:
        dispatcher.close(timeout=min(deadline_seconds, 30))
        raise
    deadline = time.monotonic() + deadline_seconds
    held: tuple[dict[str, Any], Sequence[Any]] | None = None
    closing = False
    wire_buffer = b""

    def emit(message: dict[str, Any]) -> None:
        outgoing.write(json.dumps(message, sort_keys=True) + "\n")
        outgoing.flush()

    try:
        emit({"state": "READY_FOR_GAME", "gpu_session_opened": False,
              **({"fake_cpu": fake_cpu} if fake_cpu is not None else {})})
        while not closing or labels.pending or held is not None:
            control()
            if time.monotonic() >= deadline:
                raise TimeoutError("finite Ceres RPC budget expired; owned state retained")
            for game_id in labels.drain():
                message = messages[game_id]
                emit({**responses.pop(game_id), "state": "GAME_DURABLE", "game_id": message["game_id"],
                      **({"unit_id": message["unit_id"], "request_id": game_id} if unit_roster else {})})
                del messages[game_id]
            if held is not None:
                message, roots = held
                try:
                    key = message["request_id"] if unit_roster else message["game_id"]
                    labels.submit(key, message["raw_sha256"], roots)
                except BufferError:
                    pass
                else:
                    messages[key] = message
                    held = None
            if not closing and held is None:
                ready, _, _ = select.select([incoming], [], [], 0 if b"\n" in wire_buffer else poll_seconds)
                if ready or b"\n" in wire_buffer:
                    if b"\n" not in wire_buffer:
                        chunk = os.read(incoming.fileno(), 4096)
                        if not chunk:
                            if wire_buffer:
                                raise ValueError("truncated Ceres RPC message")
                            wire_buffer = b'{"op":"stop"}\n'
                        else:
                            wire_buffer += chunk
                            if len(wire_buffer) > 8192:
                                raise ValueError("Ceres RPC wire buffer exceeds bound")
                    if b"\n" not in wire_buffer:
                        continue
                    line, wire_buffer = wire_buffer.split(b"\n", 1)
                    message = json.loads(line)
                    if message["op"] == "stop":
                        closing = True
                        dispatcher.flush()
                    elif message["op"] == "game":
                        game_id = message["game_id"]
                        if unit_roster is not None:
                            unit = message.get("unit_id")
                            if unit not in unit_roster or type(game_id) is not int or not 0 <= game_id < 128:
                                raise ValueError("exact local unit/game namespace required")
                            request = unit_roster.index(unit) * 128 + game_id
                            if message.get("request_id") != request:
                                raise ValueError("Ceres wire namespace changed")
                            game_id = request
                        if (type(game_id) is not int or not 0 <= game_id < total_games
                                or game_id in labels.pending or game_id in labels.durable):
                            raise ValueError("distinct finite Ceres game RPC required")
                        held = message, load_roots(message)
                    else:
                        raise ValueError("unsupported Ceres game RPC operation")
            else:
                time.sleep(min(poll_seconds, 0.01))
        result = {"status": "C3_GAME_LABEL_SERVICE_COMPLETE_NOT_ADMISSION",
                  "games": len(labels.durable), "logical_batch_histogram": dispatcher.histogram}
        emit(result)
        return result
    finally:
        try:
            dispatcher.close(timeout=min(deadline_seconds, 30))
        finally:
            labels.close(timeout=min(deadline_seconds, 30))


def main() -> None:
    """Retained Companion-compatible child; owner keeps custody and authority.

    Real256/512 remain rejected until a reviewed shape-qualified loader exists.
    The fixed32 real loader, provider warmups and private caches are reused.
    CPU fake execution tests the same child/RPC/publication path without ORT.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-fd", type=int, default=-1)
    parser.add_argument("--service-output")
    parser.add_argument("--fake-cpu", action="store_true")
    args = parser.parse_args()
    if args.fake_cpu and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("fake CPU service requires CUDA_VISIBLE_DEVICES=-1 before setup")
    with Path(args.config).open("rb") as stream:
        raw_config = stream.read(2 * 1024 ** 2 + 1)
    if len(raw_config) > 2 * 1024 ** 2:
        raise ValueError("bounded immutable service configuration required")
    config = json.loads(raw_config)
    expected_config = os.environ.get("DEEPFIN_SHARED_CONFIG_CANONICAL_SHA256")
    if expected_config is not None and hashlib.sha256(json.dumps(config, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest() != expected_config:
        raise ValueError("frozen owner service configuration changed before setup")
    shared = config["shared_service"]
    required = {"units", "max_games", "target_rows", "max_rows", "batch_wait_ms", "poll_seconds",
                "deadline_seconds", "physical_batch", "source_pins", "retained_service_source", "publisher_source"}
    if set(shared) != required:
        raise ValueError("closed shared service configuration required")
    seconds = shared["deadline_seconds"]
    if not math.isfinite(seconds) or not 0 < seconds <= 600:
        raise ValueError("finite inherited service setup deadline required")
    deadline = time.monotonic() + seconds
    inherited = os.environ.get("DEEPFIN_COMPARE_DEADLINE")
    if inherited is not None:
        parent_deadline = float(inherited)
        if not math.isfinite(parent_deadline) or parent_deadline <= time.monotonic():
            raise TimeoutError("inherited service deadline expired before setup")
        deadline = min(deadline, parent_deadline)
    if not args.fake_cpu and shared["physical_batch"] != 32:
        raise ValueError("real256/512 require independent shape-qualified owner loader; unarmed")
    for relative in ("chess_anti_engine/teacher_dispatch.py", "scripts/ceres_shared_game_service.py",
                     "scripts/ceres_raw_backend.py", "scripts/shared_teacher_generation.py"):
        path = Path(__file__).resolve().parents[1] / relative
        if shared["source_pins"].get(relative) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("shared service consequential source closure changed")
    def source(name: str, reference: dict[str, str], key: str) -> Any:
        path = Path(reference["path"])
        if reference["sha256"] != RETAINED_SHA[key] or hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError("retained service source pin changed")
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    publisher = source("ceres_atomic_publication_v1", shared["publisher_source"], "publisher")
    retained = source("shared_retained_ceres_service", shared["retained_service_source"], "retained")
    comparison = source("shared_retained_comparison", config["v9_source"], "comparison")
    comparison.DEADLINE = deadline
    plan = config["qualified_plan"]
    for path, digest in plan["source_pins"].items():
        if comparison.sha(path) != digest:
            raise ValueError("qualified Ceres source closure changed")
    units = {row["unit_id"]: Path(row["raw_root"]) for row in shared["units"]}
    if len(units) != len(shared["units"]) or any(set(row) not in (
            {"unit_id", "raw_root"}, {"unit_id", "raw_root", "prior_roots"}) for row in shared["units"]):
        raise ValueError("distinct closed service unit roster required")
    authorized_roots = {row["unit_id"]: (*tuple(Path(path) for path in row.get("prior_roots", [])), Path(row["raw_root"])) for row in shared["units"]}
    out = Path(args.service_output or config["service_output"])
    if out.resolve() != out or out.exists():
        raise ValueError("fresh private service output required")
    if not args.fake_cpu:
        bank = Path(config["approved_bank"])
        if not out.is_relative_to(bank) or out.name != "ceres_service" or any(not path.is_relative_to(bank) for path in units.values()):
            raise ValueError("exact approved owner service/raw bank required")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise ValueError("inherited owner CUDA visibility required")
        comparison.guard(args, plan)
        if comparison.sha(plan["models"]["ceres"]["path"]) != plan["models"]["ceres"]["sha256"]:
            raise ValueError("actual immutable Ceres model pin changed")
    out.mkdir(mode=0o700, parents=False)
    comparison.private_caches(out)
    api = comparison.imports("ceres")
    accounting: dict[str, int] = {}
    provider = None
    session = None
    bound = None
    def control() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("inherited service setup/inference deadline expired")
        if not args.fake_cpu:
            comparison.guard(args, plan)
    control()
    def infer(roots: Sequence[Any]) -> Sequence[Any]:
        nonlocal session, provider, bound
        control()
        if session is None:
            if args.fake_cpu:
                session = comparison.TimedSession(comparison.Fake(api[0], "ceres"))
                proof_fn = None
            else:
                session, _old_evaluator, proof_fn = comparison.real_eval("ceres", plan, out, api)
            for _ in range(3):
                control()
                comparison.c3_call(session, api, roots[:32])
            if proof_fn is not None:
                profile = Path(session.end_profiling())
                proof = proof_fn(json.loads(profile.read_text()))
                proof_path = out / "PROVIDER.json"
                publisher.durable_json(proof_path, {"proof": proof, "profile_sha256": comparison.sha(profile),
                    "providers": session.get_providers(), "options": session.get_provider_options(),
                    "scope": "three fixed32 warmups; same persistent shared label session", "warmup_calls": 3})
                provider = {"path": str(proof_path), "sha256": comparison.sha(proof_path)}
            session.calls.clear()
            bound = bind_ceres_raw_backend(session=session, physical_batch=shared["physical_batch"],
                gather_context=api[-2].ceres_tpg_gather_context, gather_indices=api[6], accounting=accounting)
        assert bound is not None
        try:
            return bound(roots)
        finally:
            session.calls.clear()  # retain aggregate validated counters, no unbounded call history
    load, publish = bind_retained_ceres_history(retained=retained, comparison=comparison, publisher=publisher,
        api=api, units=units, model=plan["models"]["ceres"], fake_cpu=args.fake_cpu,
        physical_batch=shared["physical_batch"], provider=lambda: provider,
        authorized_raw_roots=authorized_roots)
    result = serve_ceres_stream(sys.stdin, sys.stdout, total_games=128 * len(units),
        max_games=shared["max_games"], target_rows=shared["target_rows"], max_rows=shared["max_rows"],
        batch_wait_ms=shared["batch_wait_ms"], poll_seconds=shared["poll_seconds"],
        deadline_seconds=max(0.001, deadline - time.monotonic()), load_roots=load, infer=infer, publish=publish,
        control=control, unit_roster=tuple(units), fake_cpu=args.fake_cpu)
    publisher.durable_json(out / "RESULT.json", {**result, "status": "CPU_FAKE_ONLY" if args.fake_cpu else result["status"],
        "successful_validated_physical_accounting": accounting, "admission_credit": 0})


if __name__ == "__main__":
    main()
