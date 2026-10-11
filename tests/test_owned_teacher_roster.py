"""Actual retained descriptor/closure/SQLite owner API with CPU session seams.

The canonical output allowlist is relocated to a temporary fixture directory.
No live owner authority, lease, model session or admission artifact is created.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import bt4_root_policy_worker as worker
from scripts.shared_teacher_owner import run_owned_roster
from scripts.ceres_pipelined_client import PipelinedCompanion
from tests.test_cross_unit_teacher_generation import fixture_module
from tests.test_ceres_owner_pipeline import retained_modules
from scripts.ceres_shared_game_service import bind_retained_ceres_history
from tests.teacher_reference_fixture import reference_sources, cpu_config


def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ...]:
    retained = reference_sources(tmp_path)
    candidate = Path(__file__).resolve().parents[1]
    overlay = tmp_path / "runtime"
    helper_names = ("phase_consumption", "comparison_priority", "bt4_inherited_owner_v2", "bt4_raw_recovery_v2", "bt4_longblock_checkpoint_v2",
                    "bt4_unit_runtime_v2", "bt4_queue_consumer_v2", "bt4_tpg_bridge_v2",
                    "bt4_queue_worker_entry_v3", "ceres_game_companion_client")
    relative_paths = [f"scripts/{name}.py" for name in helper_names]
    relative_paths += ["scripts/dual_companion_config.json"]
    relative_paths += ["scripts/shared_teacher_owner.py", "scripts/shared_teacher_generation.py",
                       "chess_anti_engine/teacher_dispatch.py", "scripts/bt4_root_policy_worker.py",
                       "scripts/bt4_generation_evaluator.py", "scripts/ceres_shared_game_service.py",
                       "scripts/ceres_pipelined_client.py", "scripts/ceres_raw_backend.py"]
    for relative in relative_paths:
        destination = overlay / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = retained / relative if Path(relative).stem in helper_names or relative.endswith("dual_companion_config.json") else candidate / relative
        shutil.copyfile(source, destination)
    (overlay / "scripts/dual_companion_config.json").write_text(json.dumps(cpu_config(retained, candidate)))
    loaded = {}
    for name in helper_names:
        qualified = "scripts." + name
        spec = importlib.util.spec_from_file_location(qualified, overlay / f"scripts/{name}.py")
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    entry, consumer, recovery = (loaded[name] for name in (
        "bt4_queue_worker_entry_v3", "bt4_queue_consumer_v2", "bt4_raw_recovery_v2"))
    priority = loaded["comparison_priority"]
    readiness = tmp_path / "cpu_readiness"
    readiness.mkdir()
    for key in ("ADMISSION", "PLAN", "PREFLIGHT", "CURRENT_PLAN", "CURRENT_PREFLIGHT",
                "STARTUP_PLAN", "STARTUP_PREFLIGHT", "PHASE_PLAN", "PHASE_PREFLIGHT"):
        monkeypatch.setattr(priority, key, readiness / (key + ".missing"))
    phase_config = readiness / "phase-config.json"
    phase_config.write_text(json.dumps({"plan": {"path": str(readiness / "plan.missing"), "sha256": "0" * 64}}))
    monkeypatch.setattr(priority, "_PHASE_CONFIG", {"path": str(phase_config),
        "sha256": hashlib.sha256(phase_config.read_bytes()).hexdigest()})
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    monkeypatch.setattr(entry, "OPERATIONS", campaign)
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    # The retained owner requires a canonical interpreter path. CI invokes a
    # venv symlink; project only this fixture module's view onto its actual
    # canonical binary, leaving pytest and subprocess interpreter selection intact.
    interpreter = Path(sys.executable).resolve(strict=True)
    monkeypatch.setattr(entry, "sys", SimpleNamespace(**{**vars(sys), "executable": str(interpreter)}))
    closure = {"schema": "bt4_runtime_closure_v1", "python": {"path": str(interpreter),
        "prefix": sys.prefix, "base_prefix": sys.base_prefix, "sha256": digest(interpreter)}, "modules": {},
        "runtime_files": {str(Path(worker._lc0_ext.__file__).resolve()): digest(Path(worker._lc0_ext.__file__))},
        "repo_sources": {relative: digest(overlay / relative) for relative in relative_paths}, "native_source_pins": {}}
    closure_path = overlay / "scripts/bt4_runtime_closure_v1.json"
    closure_path.write_text(json.dumps(closure))
    pins = {relative: digest(overlay / relative) for relative in relative_paths}
    pins["scripts/bt4_runtime_closure_v1.json"] = digest(closure_path)
    fixture = fixture_module()
    fixture.MODEL_SHA = loaded["bt4_tpg_bridge_v2"].MODEL_SHA
    original_spec = fixture.spec(tmp_path)
    generation = {"parallel_games": 16, "temperature": 0, "initial_fen": fixture.SEVEN,
                      "syzygy_path": original_spec.syzygy_path}
    limits = {"max_attempts": 2, "unit_seconds": 30, "unit_io_bytes": 1000000, "unit_output_bytes": 1000000,
                  "total_seconds": 60, "total_io_bytes": 2000000, "total_output_bytes": 2000000}
    descriptors = {}
    for ordinal in (7, 9):
        unit_root = campaign / f"unit-{ordinal:012d}"
        (unit_root / "attempt-00").mkdir(parents=True)
        bindings = {"model_sha256": fixture.MODEL_SHA, "source_pins": pins, "generation": generation, "seed_base": 100}
        descriptors[f"u{ordinal}"] = {"schema_version": 3, "boot_id": entry.birth(os.getpid())["boot_id"],
            "unit": ordinal, "seed_base": 100, "output_root": str(unit_root / "attempt-00"),
            "checkpoint_path": str(unit_root / "checkpoint.sqlite"), "bindings": bindings, "checkpoint_limits": limits,
            "owner_limits": {}, "source_pins": pins, "stop_paths": [str(campaign / "STOP")], "yield_paths": [str(campaign / "YIELD")],
            "prior_roots": [], "generation": generation, "external_root": str(campaign), "attempt": 0,
            "scope_sha256": "a" * 64, "block_seconds": 30}
    seen = []
    guard = SimpleNamespace(output_root=campaign, stop_paths=(campaign / "STOP",), charge=lambda _: None,
                            limits={"deadline_unix": time.time() + 120})
    def check() -> None:
        if any(path.exists() for path in guard.stop_paths):
            raise RuntimeError("owner STOP/drain")
    guard.check = check
    def custody(*_args: Any, **kwargs: Any) -> Any:
        seen.append(("custody", kwargs["custody_profile"]))
        return lambda: os.open(os.devnull, os.O_RDONLY)  # explicit CPU custody seam
    monkeypatch.setattr(consumer, "canonical_gpu_lock_consumer", custody)
    evaluator = fixture.FakeEvaluator()
    def startup(*_args: Any) -> tuple[Any, ...]:
        seen.append(("startup",))
        for value in descriptors.values():
            with sqlite3.connect(value["checkpoint_path"]) as db:
                assert db.execute("SELECT state FROM attempts ORDER BY id DESC LIMIT 1").fetchone() == ("RUNNING",)
        return evaluator, object(), {}
    monkeypatch.setattr(consumer, "fixed_bt4_startup", startup)
    monkeypatch.setattr(worker, "verify_cuda_model_schema", lambda *_a, **_kw: None)
    monkeypatch.setattr(worker, "verify_cuda_session", lambda *_a, **_kw: (
        ("CUDAExecutionProvider", "CPUExecutionProvider"), {"device_id": "0", "gpu_mem_limit": str(2 * 1024**3),
          "cudnn_conv_algo_search": "DEFAULT", "cudnn_conv_use_max_workspace": "0"}))
    monkeypatch.setattr(worker, "CudaQualifiedEvaluator", lambda *_a, **_kw: evaluator)
    handle = fixture.fake_tablebase()
    handle.close = lambda: None
    monkeypatch.setattr(worker.tablebase, "open_strict_match_tablebase", lambda *_a, **_kw: handle)
    original_pool = worker.run_pooled_units
    def cpu_spec(spec: Any) -> Any:
        return replace(spec, requested_provider="cpu", gpu_mem_gb=0, providers=("CPUExecutionProvider",),
                       provider_options={}, cudnn_conv_algo_search=None, cudnn_conv_use_max_workspace=None)
    def pool(units: Any, actor: Any, out: Path, **kwargs: Any) -> dict[str, Any]:
        return original_pool({name: (cpu_spec(spec), tb) for name, (spec, tb) in units.items()}, actor, out, **kwargs)
    monkeypatch.setattr(worker, "run_pooled_units", pool)
    resume = worker.validate_resume_contract
    monkeypatch.setattr(worker, "validate_resume_contract", lambda spec: resume(cpu_spec(spec)))
    worker.rep_fix.apply(True, boards_discarded=True)
    options = {"entry": entry, "consumer": consumer, "recovery": recovery, "guard": guard, "parent": {}, "leases": {}, "priority_module": priority,
        "out": campaign / "controller-00", "authorize": lambda roster, _out: seen.append(("authorize", tuple(roster))),
        "max_units": 2, "max_live_games": 32, "target_rows": 8, "max_rows": 32, "batch_wait_ms": 1, "max_writes": 2,
        "deadline_seconds": 30}
    return descriptors, options, seen, evaluator, loaded, closure_path


def test_actual_owner_descriptor_runtime_checkpoint_and_finite_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors, options, seen, evaluator, _loaded, _closure = setup(tmp_path, monkeypatch)
    def authorize(roster: Any, _out: Path) -> None:
        seen.append(("authorize", tuple(roster)))
        roster.clear()  # cannot change the already validated internal snapshot
    options["authorize"] = authorize
    result = run_owned_roster(descriptors, **options)
    assert result["status"] == "COMPLETE_NOT_ADMISSION"
    assert seen[:3] == [("authorize", ("u7", "u9")), ("custody", "all_three"), ("startup",)]
    assert evaluator.calls > 0
    assert [len(values) for values in result["unit_receipts"].values()] == [128, 128]
    for value in descriptors.values():
        with sqlite3.connect(value["checkpoint_path"]) as db:
            assert db.execute("SELECT COUNT(*) FROM games").fetchone() == (128,)
            assert db.execute("SELECT state FROM attempts").fetchone() == ("RAW_COMPLETE_PENDING_ADMISSION",)


def next_attempt(descriptors: Any, options: Any) -> None:
    for value in descriptors.values():
        previous = Path(value["output_root"])
        value["attempt"] = 1
        value["prior_roots"] = [str(previous / "raw")]
        current = previous.with_name("attempt-01")
        current.mkdir()
        value["output_root"] = str(current)
    options["out"] = options["out"].with_name("controller-01")


def test_retained_owner_committed_restart_skips_every_local_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors, options, _seen, evaluator, _loaded, _closure = setup(tmp_path, monkeypatch)
    run_owned_roster(descriptors, **options)
    calls = evaluator.calls
    next_attempt(descriptors, options)
    result = run_owned_roster(descriptors, **options)
    assert result["status"] == "COMPLETE_NOT_ADMISSION"
    assert evaluator.calls == calls
    assert all(not list((Path(value["output_root"]) / "raw/games").glob("*.npz"))
               for value in descriptors.values())


def test_retained_owner_raw_before_checkpoint_ack_recovers_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors, options, _seen, _evaluator, _loaded, _closure = setup(tmp_path, monkeypatch)
    def fail_ack(_name: str, _receipt: Any) -> None:
        raise RuntimeError("fixture publication-before-owner-ack death")
    options["verify_companion"] = fail_ack  # CPU fault at the actual record_game boundary
    with pytest.raises(RuntimeError, match="before-owner-ack"):
        run_owned_roster(descriptors, **options)
    orphan_files = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                    for value in descriptors.values() for path in (Path(value["output_root"]) / "raw/games").glob("*.npz")}
    assert orphan_files
    next_attempt(descriptors, options)
    options["verify_companion"] = None
    result = run_owned_roster(descriptors, **options)
    assert result["status"] == "COMPLETE_NOT_ADMISSION"
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in orphan_files.items())
    for path in orphan_files:
        assert not (path.parent.parent.parent.with_name("attempt-01") / "raw/games" / path.name).exists()


def test_actual_owned_dual_roster_uses_retained_child_and_controller_sqlite_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    poll_depth = 0
    original_poll = PipelinedCompanion.poll
    def poll(pipeline: Any) -> None:
        nonlocal poll_depth
        assert poll_depth == 0  # burst acknowledgments never recursively poll from Future callbacks
        poll_depth += 1
        try:
            original_poll(pipeline)
        finally:
            poll_depth -= 1
    monkeypatch.setattr(PipelinedCompanion, "poll", poll)
    descriptors, options, _seen, evaluator, loaded, _closure = setup(tmp_path, monkeypatch)
    candidate = Path(__file__).resolve().parents[1]
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    reference = reference_sources(tmp_path)
    config = json.loads((Path(options["entry"].__file__).parent / "dual_companion_config.json").read_text())
    service = candidate / "scripts/ceres_shared_game_service.py"
    config.update(python=sys.executable, service_source={"path": str(service), "sha256": digest(service)},
                  service_output=str(options["out"].parent / "ceres_service"))
    config["shared_service"] = {"units": [{"unit_id": name, "raw_root": str(Path(value["output_root"]) / "raw"),
        "prior_roots": value["prior_roots"]} for name, value in descriptors.items()], "max_games": 4,
        "target_rows": 4, "max_rows": 400, "batch_wait_ms": 1, "poll_seconds": 0.001, "deadline_seconds": 60, "physical_batch": 512,
        "source_pins": {relative: digest(candidate / relative) for relative in (
            "chess_anti_engine/teacher_dispatch.py", "scripts/ceres_shared_game_service.py",
            "scripts/ceres_raw_backend.py", "scripts/shared_teacher_generation.py")},
        "retained_service_source": {"path": str(reference / "scripts/ceres_dynamic_game_service.py"),
            "sha256": digest(reference / "scripts/ceres_dynamic_game_service.py")},
        "publisher_source": {"path": str(reference / "scripts/ceres_atomic_publication_v1.py"),
            "sha256": digest(reference / "scripts/ceres_atomic_publication_v1.py")}}
    config_path = tmp_path / "attempt-companion-config.json"
    config_path.write_text(json.dumps(config))
    owner_thread = __import__("threading").get_ident()
    retained, comparison, publisher, api = retained_modules(monkeypatch, tmp_path)
    def history_verifier() -> Any:
        units = {name: Path(value["output_root"]) / "raw" for name, value in descriptors.items()}
        load, publish = bind_retained_ceres_history(retained=retained, comparison=comparison, publisher=publisher,
            api=api, units=units, model=comparison.MODELS["ceres"], fake_cpu=True,
            physical_batch=512, provider=None, authorized_raw_roots={name:
                (*tuple(Path(path) for path in value["prior_roots"]), units[name])
                for name, value in descriptors.items()})
        return load, publish
    load, publish = history_verifier()
    def verify(_name: str, receipt: Any) -> None:
        assert __import__("threading").get_ident() == owner_thread
        ref = receipt["ceres_companion"]
        path = Path(ref["path"])
        assert digest(path) == ref["sha256"]
        companion = json.loads(path.read_text())
        assert companion["raw_sha256"] == receipt["sha256"]
        assert companion["real_rows"] == receipt["rows"]
        message = {"unit_id": _name, "game_id": companion["game_id"],
            "request_id": tuple(descriptors).index(_name) * 128 + companion["game_id"],
            "raw_path": receipt["path"], "raw_sha256": receipt["sha256"]}
        assert load(message) == ()  # literal retained history + complete NPZ/identity verifier
        assert publish(message, ())["companion"] == ref
        with np.load(companion["labels"]["path"], allow_pickle=False) as archive:
            assert archive["value_logits"].tolist() == [[1, 2, 3]]
            assert archive["value2_logits"].tolist() == [[4, 5, 6]]
    options.update(companion_module=loaded["ceres_game_companion_client"], companion_config=config_path,
                   fake_cpu=True, verify_companion=verify, max_pending=4)
    result = run_owned_roster(descriptors, **options)
    assert result["status"] == "COMPLETE_NOT_ADMISSION"
    assert all(len(receipts) == 128 and all("ceres_companion" in receipt for receipt in receipts)
               for receipts in result["unit_receipts"].values())
    service_result = json.loads((options["out"].parent / "ceres_service/RESULT.json").read_text())
    assert service_result["status"] == "CPU_FAKE_ONLY"
    assert service_result["successful_validated_physical_accounting"]["real_rows"] == 256
    calls = evaluator.calls
    next_attempt(descriptors, options)
    config["shared_service"]["units"] = [{"unit_id": name,
        "raw_root": str(Path(value["output_root"]) / "raw"), "prior_roots": value["prior_roots"]}
        for name, value in descriptors.items()]
    second_service = options["out"].parent / "service-attempt-01"
    second_service.mkdir()
    config["service_output"] = str(second_service / "ceres_service")
    config_path.write_text(json.dumps(config))  # deployment paths change; science-bound template stays exact
    load, publish = history_verifier()
    result = run_owned_roster(descriptors, **options)
    assert result["status"] == "COMPLETE_NOT_ADMISSION"
    assert evaluator.calls == calls
    service_result = json.loads((second_service / "ceres_service/RESULT.json").read_text())
    assert service_result["successful_validated_physical_accounting"].get("calls", 0) == 0


@pytest.mark.parametrize("failure", ["geometry", "roster_geometry", "scope", "external_root", "duplicate", "source", "native", "stop_coverage", "authorize", "yield"])
def test_owner_preflight_rejects_or_defers_before_custody_and_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    descriptors, options, seen, _evaluator, _loaded, closure = setup(tmp_path, monkeypatch)
    if failure == "geometry":
        options["max_rows"] = 0
    elif failure == "roster_geometry":
        template = descriptors["u7"]
        descriptors = {f"u{index}": dict(template, unit=index) for index in range(4097)}
        options.update(max_units=4097, max_live_games=4097 * 64)
    elif failure == "scope":
        descriptors["u9"]["scope_sha256"] = "b" * 64
    elif failure == "external_root":
        foreign = options["out"].parent / "foreign-campaign"
        root = foreign / "unit-000000000009/attempt-00"
        root.mkdir(parents=True)
        descriptors["u9"].update(external_root=str(foreign), output_root=str(root),
                                checkpoint_path=str(root.parent / "checkpoint.sqlite"))
    elif failure == "duplicate":
        descriptors["u9"]["unit"] = 7
    elif failure == "source":
        descriptors["u7"]["source_pins"]["scripts/shared_teacher_owner.py"] = "0" * 64
    elif failure == "native":
        record = json.loads(closure.read_text())
        key = next(iter(record["runtime_files"]))
        record["runtime_files"][key] = "0" * 64
        closure.write_text(json.dumps(record))
        for value in descriptors.values():
            value["source_pins"]["scripts/bt4_runtime_closure_v1.json"] = hashlib.sha256(closure.read_bytes()).hexdigest()
    elif failure == "stop_coverage":
        options["guard"].stop_paths = ()
    elif failure == "authorize":
        def reject(*_args: Any) -> None:
            raise ValueError("no explicit whole-roster authority")
        options["authorize"] = reject
    elif failure == "yield":
        Path(descriptors["u7"]["yield_paths"][0]).write_text("CPU fixture")
    if failure == "yield":
        assert run_owned_roster(descriptors, **options)["status"] == "PRIORITY_DEFERRED_BEFORE_CUDA"
    else:
        message = {"geometry": "geometry", "roster_geometry": "roster", "scope": "scope", "external_root": "scope", "duplicate": "roster",
                   "source": "source pin", "native": "runtime file pin", "stop_coverage": "STOP",
                   "authorize": "authority"}[failure]
        with pytest.raises(ValueError, match=message):
            run_owned_roster(descriptors, **options)
    assert seen == []
    assert all(not (Path(value["output_root"]) / "raw").exists() for value in descriptors.values())


@pytest.mark.parametrize("priority_kind", ["marker", "training_readiness"])
def test_bt4_only_poststartup_priority_retains_attempt_without_new_moves_or_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, priority_kind: str,
) -> None:
    descriptors, options, seen, evaluator, _loaded, _closure = setup(tmp_path, monkeypatch)
    original = evaluator.evaluate_roots
    applied = []
    def evaluate(boards: Any, x_batch: Any) -> Any:
        result = original(boards, x_batch)
        if priority_kind == "marker":
            Path(descriptors["u7"]["yield_paths"][0]).touch()
        else:
            monkeypatch.setattr(options["priority_module"], "training_ready", lambda: True)
        return result
    monkeypatch.setattr(evaluator, "evaluate_roots", evaluate)
    monkeypatch.setattr(worker.BT4RootPolicyStepper, "apply_root_outputs", lambda *_args: applied.append(True))
    with pytest.raises(RuntimeError, match=r"priority|dispatcher failed") as failure:
        run_owned_roster(descriptors, **options)
    causes = []
    error: BaseException | None = failure.value
    while error is not None:
        causes.append(str(error))
        error = error.__cause__
    assert any("owner priority yield/drain" in cause for cause in causes)
    assert seen[:3] == [("authorize", ("u7", "u9")), ("custody", "all_three"), ("startup",)]
    assert evaluator.calls == 1
    assert applied == []
    for descriptor in descriptors.values():
        assert list(Path(descriptor["output_root"]).rglob("game_*.npz")) == []
        with sqlite3.connect(descriptor["checkpoint_path"]) as db:
            assert db.execute("SELECT COUNT(*) FROM games").fetchone() == (0,)
            assert db.execute("SELECT state FROM attempts").fetchone() == ("FAILED_RETAINED_ZERO_CREDIT",)


def test_materialized_reference_drift_is_rejected_without_overwrite(tmp_path: Path) -> None:
    reference = reference_sources(tmp_path)
    path = reference / "scripts/ceres_game_companion_client.py"
    changed = path.read_bytes() + b"\n# owned negative fixture\n"
    path.write_bytes(changed)
    with pytest.raises(ValueError, match="materialized retained reference changed"):
        reference_sources(tmp_path)
    assert path.read_bytes() == changed


@pytest.mark.parametrize("fault", [None, "interpreter_hash", "native_alias"])
def test_retained_canonical_closure_with_fixture_interpreter_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str | None,
) -> None:
    actual = Path(sys.executable).resolve(strict=True)
    alias = tmp_path / "python-alias"
    alias.symlink_to(actual)
    monkeypatch.setattr(sys, "executable", str(alias))
    _descriptors, _options, seen, _evaluator, loaded, closure_path = setup(tmp_path, monkeypatch)
    entry = loaded["bt4_queue_worker_entry_v3"]
    assert sys.executable == str(alias)
    assert entry.sys.executable == str(actual)
    closure = json.loads(closure_path.read_text())
    assert closure["python"]["path"] == str(actual)
    if fault == "interpreter_hash":
        closure["python"]["sha256"] = "0" * 64
    elif fault == "native_alias":
        path, digest = next(iter(closure["runtime_files"].items()))
        native_alias = tmp_path / "native-alias"
        native_alias.symlink_to(path)
        closure["runtime_files"] = {str(native_alias): digest}
    closure_path.write_text(json.dumps(closure))
    if fault is None:
        entry.verify_runtime_closure()
    else:
        with pytest.raises(ValueError, match=r"runtime file pin differs|runtime canonical file differs"):
            entry.verify_runtime_closure()
    assert seen == []


def test_reference_manifest_distinguishes_original_and_neutral_path_bytes(tmp_path: Path) -> None:
    fixtures = Path(__file__).with_name("fixtures") / "teacher_reference"
    manifest = json.loads((fixtures / "manifest.json").read_text())
    assert manifest["schema"] == 2
    changed = [row for row in manifest["files"] if row["sha256"] != row["original_sha256"]]
    assert len(changed) == 6
    assert sum(row["path_relocations"]["home"] for row in changed) == 26
    assert sum(row["path_relocations"]["workspace"] for row in changed) == 1
    reference = reference_sources(tmp_path)
    for row in manifest["files"]:
        raw = (reference / row["relative"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        assert len(raw) == row["bytes"]
        if row not in changed:
            assert row["sha256"] == row["original_sha256"]
            assert row["bytes"] == row["original_bytes"]


@pytest.mark.parametrize("alias", ["_phase_requested", "_checked", "_ready_refs"])
def test_owner_rejects_differently_loaded_priority_phase_callable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str,
) -> None:
    descriptors, options, seen, _evaluator, _loaded, _closure = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(options["priority_module"], alias, lambda *_args: None)
    with pytest.raises(ValueError, match="loaded priority phase helper source differs"):
        run_owned_roster(descriptors, **options)
    assert seen == []
    assert all(not (Path(value["output_root"]) / "raw").exists() for value in descriptors.values())


def test_poststartup_yield_after_committed_game_recovers_without_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptors, options, _seen, _evaluator, loaded, _closure = setup(tmp_path, monkeypatch)
    checkpoint_type = loaded["bt4_longblock_checkpoint_v2"].Checkpoint
    original = checkpoint_type.record_game
    marker = Path(next(iter(descriptors.values()))["yield_paths"][0])
    committed = []

    def record(checkpoint: Any, attempt: int, game: int, receipt: Any) -> Any:
        result = original(checkpoint, attempt, game, receipt)
        if result and not committed:
            committed.append((game, receipt))
            marker.touch()
        return result

    monkeypatch.setattr(checkpoint_type, "record_game", record)
    with pytest.raises(RuntimeError, match="priority yield/drain"):
        run_owned_roster(descriptors, **options)
    assert len(committed) == 1
    game, receipt = committed[0]
    path = Path(receipt["path"])
    old_bytes = path.read_bytes()
    assert hashlib.sha256(old_bytes).hexdigest() == receipt["sha256"]
    assert receipt["canonical_raw"]["game_id"] == game
    total = 0
    for descriptor in descriptors.values():
        with sqlite3.connect(descriptor["checkpoint_path"]) as db:
            total += db.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            assert db.execute("SELECT state FROM attempts").fetchone() == ("FAILED_RETAINED_ZERO_CREDIT",)
    assert total == 1

    monkeypatch.setattr(checkpoint_type, "record_game", original)
    marker.unlink()  # only this fixture's owned marker
    next_attempt(descriptors, options)
    result = run_owned_roster(descriptors, **options)
    assert result["status"] == "COMPLETE_NOT_ADMISSION"
    assert path.read_bytes() == old_bytes  # exact stored history, targets, RNG/provenance metadata
    owner_name = next(name for name, value in descriptors.items() if path.is_relative_to(Path(value["prior_roots"][0])))
    owner = descriptors[owner_name]
    assert not (Path(owner["output_root"]) / "raw/games" / path.name).exists()
    for descriptor in descriptors.values():
        with sqlite3.connect(descriptor["checkpoint_path"]) as db:
            assert db.execute("SELECT COUNT(*) FROM games").fetchone() == (128,)
    retained_receipts = [value for value in result["unit_receipts"][owner_name]
                         if value["game_id"] == game and value["sha256"] == receipt["sha256"]]
    assert len(retained_receipts) == 1


@pytest.mark.parametrize("relative", ["scripts/shared_teacher_owner.py", "scripts/shared_teacher_generation.py",
                                     "chess_anti_engine/teacher_dispatch.py", "scripts/ceres_pipelined_client.py"])
def test_owner_rejects_pinned_overlay_different_from_executing_shared_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str,
) -> None:
    descriptors, options, seen, _evaluator, loaded, _closure = setup(tmp_path, monkeypatch)
    path = Path(options["entry"].__file__).parents[1] / relative
    path.write_text(path.read_text() + "\n# Different explicitly pinned overlay bytes.\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for descriptor in descriptors.values():
        descriptor["source_pins"][relative] = digest
    if relative == "scripts/ceres_pipelined_client.py":
        options.update(companion_module=loaded["ceres_game_companion_client"],
                       companion_config=tmp_path / "not-read.json", verify_companion=lambda *_args: None)
    with pytest.raises(ValueError, match="loaded shared owner/controller"):
        run_owned_roster(descriptors, **options)
    assert seen == []
    assert all(not (Path(value["output_root"]) / "raw").exists() for value in descriptors.values())
