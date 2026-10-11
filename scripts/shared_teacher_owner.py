"""Checkpoint acknowledgments beneath the existing admitted queue owner.

The retained queue owner supplies roster authority and inherited custody. Its
Checkpoint and immutable raw verifier stay authoritative. Controller callbacks
finish SQLite transactions only after the durable Ceres acknowledgment.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import os
import threading
import time
from typing import Any


def bind_owner_checkpoints(
    *, units: Mapping[str, tuple[int, Path, Any, int]],
    verify_raw: Callable[[str, Path, int], dict[str, Any]],
    verify_completed: Callable[[str, dict[str, Any]], None],
    companion: Any | None = None,
    authorized_raw_roots: Mapping[str, tuple[Path, ...]] | None = None,
) -> tuple[dict[str, dict[int, dict[str, Any]]],
           Callable[[str, int, dict[str, Any]], dict[str, Any] | Future[dict[str, Any]]]]:
    """Bind already-started per-unit attempts to the actual finite controller.

    Values are (ordinal, fresh raw directory, retained Checkpoint, attempt ID).
    begin() must run before raw directories are created; the owner caller retains
    custody, source/roster authorization, guards, STOP and final finish()/close().
    Failed futures remain fail-closed until owner restart, without in-place retry.
    Prior roots must come from the validated per-unit descriptor; without them,
    only this exact attempt's raw root is accepted during receipt restoration.
    """
    units = dict(units)
    roots = ({name: (value[1],) for name, value in units.items()} if authorized_raw_roots is None
             else {name: tuple(values) for name, values in authorized_raw_roots.items()})
    if set(roots) != set(units) or any(units[name][1] not in values or not 1 <= len(values) <= 8
                                     or len(set(values)) != len(values)
                                     or any(path.resolve() != path or not path.is_absolute() for path in values)
                                     for name, values in roots.items()):
        raise ValueError("exact finite authorized per-unit raw roots required")
    flat_roots = [path for values in roots.values() for path in values]
    if len(set(flat_roots)) != len(flat_roots):
        raise ValueError("authorized roots overlap across units")
    owner_thread = threading.get_ident()
    completed: dict[str, dict[int, dict[str, Any]]] = {}
    if (not 1 <= len(units) <= 4096
            or len({value[0] for value in units.values()}) != len(units)
            or len({value[1] for value in units.values()}) != len(units)
            or len({value[2].path for value in units.values()}) != len(units)):
        raise ValueError("distinct finite per-unit checkpoint ownership required")

    def owner_only() -> None:
        if threading.get_ident() != owner_thread:
            raise RuntimeError("owner SQLite acknowledgment requires controller thread")

    def canonical_raw(raw: dict[str, Any], actual: dict[str, Any], path: Path, game: int) -> None:
        if (set(raw) != {"path", "sha256", "game_id", "status", "rows", "discarded_rows", "launch_sha256"}
                or type(game) is not int or not 0 <= game < 128 or raw["game_id"] != game
                or path.name != f"game_{game:08d}.npz" or raw["path"] != path.name
                or any(raw[key] != actual[key] for key in ("sha256", "status", "rows"))):
            raise ValueError("canonical owner raw receipt changed")
        launch = path.parent.parent / "launch.json"
        if (launch.resolve() != launch or launch.stat().st_size > 2 * 1024 ** 2
                or hashlib.sha256(launch.read_bytes()).hexdigest() != raw["launch_sha256"]):
            raise ValueError("canonical owner launch binding changed")
        # The bounded retained verifier already checked this exact immutable NPZ.
        import numpy as np
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(archive["metadata"].tobytes().decode())
        if metadata["game_id"] != game or metadata["discarded_rows"] != raw["discarded_rows"]:
            raise ValueError("canonical owner embedded game accounting changed")

    for name, (ordinal, root, checkpoint, attempt) in units.items():
        owner_only()
        if (checkpoint.db.execute("SELECT unit,state,raw_root FROM attempts WHERE id=?", (attempt,)).fetchone()
                != (ordinal, "RUNNING", str(root))):
            raise ValueError("exact active unit/attempt/raw binding required")
        receipts: dict[int, dict[str, Any]] = {}
        def verify(receipt: dict[str, Any], unit: str = name,
                   target: dict[int, dict[str, Any]] = receipts) -> None:
            canonical = receipt["canonical_raw"]
            game = canonical["game_id"]
            path = Path(receipt["path"])
            if path not in tuple(root / "games" / f"game_{game:08d}.npz" for root in roots[unit]):
                raise ValueError("restored raw path is outside exact unit attempt roots")
            actual = verify_raw(unit, path, game)
            if any(receipt.get(key) != value for key, value in actual.items()):
                raise ValueError("retained owner raw binding changed")
            canonical_raw(canonical, actual, Path(receipt["path"]), game)
            verify_completed(unit, receipt)
            target[game] = {**canonical, **({"ceres_companion": receipt["ceres_companion"]}
                                            if "ceres_companion" in receipt else {})}
        database_games = checkpoint.completed(ordinal, verify)
        if tuple(sorted(receipts)) != database_games:
            raise ValueError("canonical receipt and checkpoint game IDs disagree")
        completed[name] = receipts

    def acknowledge(unit: str, game: int, raw: dict[str, Any]) -> dict[str, Any] | Future[dict[str, Any]]:
        owner_only()
        _ordinal, root, checkpoint, attempt = units[unit]
        path = root / "games" / raw["path"]
        verified = verify_raw(unit, path, game)
        canonical_raw(raw, verified, path, game)

        def commit(combined: dict[str, Any]) -> dict[str, Any]:
            owner_only()
            if any(combined.get(key) != value for key, value in verified.items()):
                raise ValueError("companion acknowledgment changed verified raw")
            owner_receipt = {**combined, "canonical_raw": dict(raw)}
            verify_completed(unit, owner_receipt)
            checkpoint.record_game(attempt, game, owner_receipt)
            return {**raw, **({"ceres_companion": combined["ceres_companion"]}
                             if "ceres_companion" in combined else {})}

        if companion is None:
            return commit(verified)
        pending = companion.submit(unit, verified, game)
        acknowledged: Future[dict[str, Any]] = Future()
        acknowledged.set_running_or_notify_cancel()
        def complete(future: Future[dict[str, Any]]) -> None:
            try:
                value = commit(future.result())
            except BaseException as exc:
                acknowledged.set_exception(exc)
            else:
                acknowledged.set_result(value)
        pending.add_done_callback(complete)
        return acknowledged

    return completed, acknowledge


def run_owned_roster(
    descriptors: Mapping[str, dict[str, Any]], *, entry: Any, consumer: Any, recovery: Any,
    guard: Any, parent: dict[str, Any], leases: dict[str, Any], out: Path, priority_module: Any,
    authorize: Callable[[Mapping[str, dict[str, Any]], Path], None],
    max_units: int, max_live_games: int, target_rows: int, max_rows: int,
    batch_wait_ms: float, max_writes: int, deadline_seconds: float,
    companion_module: Any | None = None, companion_config: Path | None = None,
    fake_cpu: bool = False,
    verify_companion: Callable[[str, dict[str, Any]], None] | None = None,
    max_pending: int = 128, wire_bytes: int = 8192,
) -> dict[str, Any]:
    """Execute a finite roster inside the existing queue's admitted owner.

    The queue caller supplies its explicit roster authorizer and aggregate guard.
    A current single-unit GO cannot authorize this roster. All descriptors/source
    closure are checked before inherited custody, checkpoint attempts or startup.
    This API creates no authorization artifact or queue item. Production adoption
    still requires an independently admitted roster and real GPU qualification.
    """
    from scripts import bt4_root_policy_worker as worker
    from scripts import shared_teacher_generation as generation_module
    from scripts import ceres_pipelined_client as companion_adapter
    from chess_anti_engine import teacher_dispatch as dispatcher_module
    from scripts.ceres_pipelined_client import PipelinedCompanion

    frozen = json.loads(json.dumps(dict(descriptors)))
    worker.validate_cross_unit_budgets(tuple(frozen), max_units=max_units, max_live_games=max_live_games,
        max_writes=max_writes, deadline_seconds=deadline_seconds, target_rows=target_rows,
        max_rows=max_rows, batch_wait_ms=batch_wait_ms)
    if companion_module is not None:
        PipelinedCompanion.validate_bounds(tuple(frozen), max_pending=max_pending, wire_bytes=wire_bytes)
    if not 1 <= len(frozen) <= min(max_units, 4096) or len({value["unit"] for value in frozen.values()}) != len(frozen):
        raise ValueError("distinct explicit finite owner roster required")
    if companion_module is not None and (verify_companion is None or companion_config is None):
        raise ValueError("dual owner requires verified durable companion recovery")
    source_root = Path(entry.__file__).resolve().parents[1]
    required_shared = ("scripts/shared_teacher_owner.py", "scripts/shared_teacher_generation.py",
                       "chess_anti_engine/teacher_dispatch.py", "scripts/bt4_root_policy_worker.py",
                       "scripts/bt4_runtime_closure_v1.json", "scripts/comparison_priority.py",
                       "scripts/phase_consumption.py")
    if companion_module is not None:
        required_shared += ("scripts/ceres_pipelined_client.py", "scripts/ceres_shared_game_service.py",
                            "scripts/ceres_raw_backend.py", "scripts/ceres_game_companion_client.py",
                            "scripts/dual_companion_config.json")
    prepared = {}
    specs = {}
    for name, descriptor in frozen.items():
        root, checkpoint_path = entry.validate_descriptor(descriptor, out)
        for relative in required_shared:
            path = source_root / relative
            if descriptor["source_pins"].get(relative) != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("shared owner source closure changed")
        prepared[name] = root, checkpoint_path
        generation = descriptor["generation"]
        spec = worker.WorkerSpec(out=root / "raw", games=128, max_plies=400,
            seed=descriptor["seed_base"] + descriptor["unit"],
            parallel_games=generation["parallel_games"], temperature=generation["temperature"],
            initial_fen=generation["initial_fen"], syzygy_path=generation["syzygy_path"],
            model_sha256=descriptor["bindings"]["model_sha256"], model_path=entry.MODEL,
            outcome_mode=worker.OUTCOME_MODE, providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            provider_options={}, requested_provider="cuda",
            gpu_mem_gb=2, cudnn_conv_algo_search="DEFAULT", cudnn_conv_use_max_workspace=0,
            research_capacity_128x400=True)
        spec.validate(check_realized_provider=False)
        specs[name] = spec
    contracts = {(value["scope_sha256"], value["seed_base"], value["external_root"],
                  json.dumps(value["source_pins"], sort_keys=True),
                  json.dumps(value["generation"], sort_keys=True)) for value in frozen.values()}
    if len(contracts) != 1 or sum(value["generation"]["parallel_games"] for value in frozen.values()) > max_live_games:
        raise ValueError("one explicit shared owner scope/source/generation and aggregate bound required")
    entry.external(out)
    campaign = Path(next(iter(frozen.values()))["external_root"])
    stops = {Path(path) for value in frozen.values() for path in value["stop_paths"]}
    if guard.output_root != campaign or not out.is_relative_to(campaign) or not stops.issubset(set(guard.stop_paths)):
        raise ValueError("aggregate owner output/STOP coverage differs")
    for module in (entry, consumer, recovery, priority_module, *((companion_module,) if companion_module is not None else ())):
        path = Path(module.__file__).resolve()
        relative = path.relative_to(source_root).as_posix()
        if any(value["source_pins"].get(relative) != hashlib.sha256(path.read_bytes()).hexdigest()
               for value in frozen.values()):
            raise ValueError("loaded retained owner source closure differs")
    if hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest() != next(iter(frozen.values()))["source_pins"]["scripts/bt4_root_policy_worker.py"]:
        raise ValueError("loaded shared worker source differs")
    phase_pin = next(iter(frozen.values()))["source_pins"]["scripts/phase_consumption.py"]
    if any(hashlib.sha256(Path(function.__code__.co_filename).read_bytes()).hexdigest() != phase_pin
           for function in (priority_module._phase_requested, priority_module._checked, priority_module._ready_refs)):
        raise ValueError("loaded priority phase helper source differs")
    loaded_shared = (("scripts/shared_teacher_owner.py", Path(__file__)),
                     ("scripts/shared_teacher_generation.py", Path(generation_module.__file__)),
                     ("chess_anti_engine/teacher_dispatch.py", Path(dispatcher_module.__file__)))
    if companion_module is not None:
        loaded_shared += (("scripts/ceres_pipelined_client.py", Path(companion_adapter.__file__)),)
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != next(iter(frozen.values()))["source_pins"][relative]
           for relative, path in loaded_shared):
        raise ValueError("loaded shared owner/controller source differs")
    config: dict[str, Any] | None = None
    if companion_module is not None:
        # The reviewed template stays science-bound across attempts. Only the
        # finite deployment paths/roster vary; putting those paths in bindings
        # would incorrectly invalidate every retained checkpoint on restart.
        template = entry.read_json(source_root / "scripts/dual_companion_config.json")
        config = entry.read_json(companion_config)
        assert config is not None
        if (any(config.get(key) != template[key] for key in ("v9_source", "qualified_plan", "cleanup_source"))
                or (not fake_cpu and config["python"] != template["python"])
                or config["service_source"]["sha256"] != next(iter(frozen.values()))["source_pins"]["scripts/ceres_shared_game_service.py"]
                or not Path(config["service_output"]).is_relative_to(campaign)):
            raise ValueError("retained owner Ceres loader/template binding changed")
        shared = config["shared_service"]
        expected_units = [{"unit_id": name, "raw_root": str(specs[name].out), "prior_roots": value["prior_roots"]}
                          for name, value in frozen.items()]
        if shared["units"] != expected_units or any(
                shared["source_pins"].get(relative) != next(iter(frozen.values()))["source_pins"].get(relative)
                for relative in ("chess_anti_engine/teacher_dispatch.py", "scripts/ceres_shared_game_service.py",
                                 "scripts/ceres_raw_backend.py", "scripts/shared_teacher_generation.py")):
            raise ValueError("exact owner Ceres roster/source configuration required")
        if not fake_cpu and shared["physical_batch"] != 32:
            raise ValueError("real256/512 owner remains unarmed pending qualification")
    entry.verify_runtime_closure()  # actual native/interpreter/source checker, before startup
    yield_paths = tuple(dict.fromkeys(path for value in frozen.values() for path in value["yield_paths"]))
    if priority_module.requested(yield_paths):
        return {"status": "PRIORITY_DEFERRED_BEFORE_CUDA", "native_admission": False}
    authorize(json.loads(json.dumps(frozen)), out)  # callback cannot mutate validated internal contracts
    owner_thread = threading.get_ident()
    guard.check()
    fd = consumer.canonical_gpu_lock_consumer(worker, parent, leases, custody_profile="all_three")()
    checkpoints: dict[str, Any] = {}
    attempts: dict[str, int] = {}
    handle = None
    client = None
    pipeline = None
    finished: set[str] = set()
    def control(*, poll: bool = True) -> None:
        guard.check()
        if priority_module.requested(yield_paths):
            raise RuntimeError("owner priority yield/drain")
        # Backend/writer guards never poll or mutate the SQLite controller.
        if poll and pipeline is not None and threading.get_ident() == owner_thread and pipeline.pending:
            pipeline.poll()
    try:
        control()
        for name, descriptor in frozen.items():
            checkpoint = entry.Checkpoint(prepared[name][1], descriptor["bindings"],
                                          descriptor["checkpoint_limits"], guard.check)
            checkpoints[name] = checkpoint
            checkpoint.recover(entry.alive)
            attempts[name] = checkpoint.begin(descriptor["unit"], entry.birth(os.getpid()), str(specs[name].out))
        evaluator, session, schema = consumer.fixed_bt4_startup(worker, out.parent, control)
        worker.verify_cuda_model_schema(session, **schema)
        providers, options = worker.verify_cuda_session(session, gpu_mem_gb=2,
            cudnn_conv_algo_search="DEFAULT", cudnn_conv_use_max_workspace=0)
        generation = next(iter(frozen.values()))["generation"]
        handle = worker.tablebase.open_strict_match_tablebase(generation["syzygy_path"], max_pieces=6)
        worker.rep_fix.apply(True, boards_discarded=True)
        units = {}
        for name in frozen:
            spec = replace(specs[name], providers=tuple(providers), provider_options=options)
            spec.validate()
            units[name] = spec, handle
        actor = worker.CudaQualifiedEvaluator(evaluator, session, out,
                                              proof_outputs=[spec.out for spec, _ in units.values()])
        if companion_module is not None:
            assert config is not None
            if entry.read_json(companion_config) != config:
                raise ValueError("owner Ceres configuration changed after preflight")
            previous = os.environ.get("DEEPFIN_SHARED_CONFIG_CANONICAL_SHA256")
            previous_deadline = os.environ.get("DEEPFIN_COMPARE_DEADLINE")
            absolute = time.monotonic() + min(deadline_seconds, guard.limits["deadline_unix"] - time.time())
            if previous_deadline is not None:
                absolute = min(absolute, float(previous_deadline))
            os.environ["DEEPFIN_COMPARE_DEADLINE"] = str(absolute)
            os.environ["DEEPFIN_SHARED_CONFIG_CANONICAL_SHA256"] = hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            try:
                client = companion_module.Companion(companion_config, fd, guard.check, guard.charge,
                    priority=lambda: priority_module.requested(yield_paths),
                    fake_cpu=fake_cpu)
            finally:
                if previous is None:
                    os.environ.pop("DEEPFIN_SHARED_CONFIG_CANONICAL_SHA256", None)
                else:
                    os.environ["DEEPFIN_SHARED_CONFIG_CANONICAL_SHA256"] = previous
                if previous_deadline is None:
                    os.environ.pop("DEEPFIN_COMPARE_DEADLINE", None)
                else:
                    os.environ["DEEPFIN_COMPARE_DEADLINE"] = previous_deadline
            pipeline = PipelinedCompanion(client, tuple(units), max_pending=max_pending, wire_bytes=wire_bytes)

        def verify_raw(name: str, path: Path, game: int) -> dict[str, Any]:
            return recovery.verified_game(path, game, checkpoints[name].binding,
                frozen[name]["bindings"]["model_sha256"], guard.check, guard.charge)

        def verify_completed(name: str, receipt: dict[str, Any]) -> None:
            control(poll=False)  # Future callbacks must not recursively poll their wire owner.
            if verify_companion is not None:
                verify_companion(name, receipt)
            control(poll=False)

        # Reuse the retained bounded128-name scanner. Prior raw/C3 publication
        # without checkpoint acknowledgment is reconciled before new actors.
        # Missing C3 labels use the same owned pipeline; startup waits are bounded
        # by the aggregate owner guard, without another scheduler or session.
        for name, (spec, _) in units.items():
            roots = [Path(path) for path in frozen[name]["prior_roots"]]
            recorded: dict[int, dict[str, Any]] = {}
            checkpoints[name].completed(frozen[name]["unit"],
                lambda receipt, target=recorded: target.setdefault(receipt["canonical_raw"]["game_id"], receipt))
            for root in roots:
                if root.exists():
                    worker.validate_resume_contract(replace(spec, out=root))
            def recover_game(path: Path, game: int, unit: str = name,
                             saved: dict[int, dict[str, Any]] = recorded) -> dict[str, Any]:
                raw = verify_raw(unit, path, game)
                if game in saved:
                    old = saved[game]
                    if any(old.get(key) != value for key, value in raw.items()):
                        raise ValueError("prior committed raw receipt changed")
                    verify_completed(unit, old)
                    return old
                import numpy as np
                with np.load(path, allow_pickle=False) as archive:
                    meta = json.loads(archive["metadata"].tobytes().decode())
                canonical = {"path": path.name, "sha256": raw["sha256"], "game_id": game,
                    "status": raw["status"], "rows": raw["rows"], "discarded_rows": meta["discarded_rows"],
                    "launch_sha256": hashlib.sha256((path.parent.parent / "launch.json").read_bytes()).hexdigest()}
                if pipeline is not None:
                    pending = pipeline.submit(unit, raw, game)
                    while not pending.done():
                        control()
                        pipeline.poll()
                    raw = pending.result()
                receipt = {**raw, "canonical_raw": canonical}
                verify_completed(unit, receipt)
                return receipt
            recovery.reconcile(checkpoints[name], attempts[name], roots, recover_game)

        completed, acknowledge = bind_owner_checkpoints(
            units={name: (frozen[name]["unit"], spec.out, checkpoints[name], attempts[name])
                   for name, (spec, _) in units.items()},
            authorized_raw_roots={name: (*tuple(Path(path) for path in frozen[name]["prior_roots"]), spec.out) for name, (spec, _) in units.items()},
            verify_raw=verify_raw, verify_completed=verify_completed, companion=pipeline)

        result = worker.run_pooled_units(units, actor, out, max_units=max_units, max_live_games=max_live_games,
            target_rows=target_rows, max_rows=max_rows, batch_wait_ms=batch_wait_ms, max_writes=max_writes,
            deadline_seconds=deadline_seconds, owner_completed=completed, acknowledge=acknowledge, control=control)
        for name, checkpoint in checkpoints.items():
            checkpoint.finish(attempts[name], result["status"] == "COMPLETE_NOT_ADMISSION")
            finished.add(name)
        return result
    finally:
        # Original Companion retains its reviewed private-session cleanup; the
        # existing queue hard reaper owns blocked native/cleanup deadlines.
        try:
            if client is not None:
                client.close()
        finally:
            try:
                close_errors: list[BaseException] = []
                for name, checkpoint in checkpoints.items():
                    try:
                        if name in attempts and name not in finished:
                            checkpoint.finish(attempts[name], False)
                    except BaseException:
                        pass  # stopped guard leaves RUNNING evidence for exact owner recovery
                    finally:
                        try:
                            checkpoint.close()
                        except BaseException as exc:
                            close_errors.append(exc)
                if handle is not None:
                    try:
                        handle.close()
                    except BaseException as exc:
                        close_errors.append(exc)
                if close_errors:
                    raise close_errors[0]
            finally:
                os.close(fd)  # validated duplicate only; inherited owner lease remains held
