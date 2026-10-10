"""CPU-only opt-in CLI and same-session proof publication contracts."""
from __future__ import annotations

import hashlib
import json
import sys
import threading
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from scripts import bt4_root_policy_worker as worker
from tests.test_cross_unit_teacher_generation import fixture_module
from tests.test_cross_unit_teacher_generation import small_units


def cli(tmp_path: Path, *, provider: str = "cpu", digest: str | None = None) -> list[str]:
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps([{"unit_id": "u0", "ordinal": 7}, {"unit_id": "u1", "ordinal": 9}]))
    model = tmp_path / "fake.onnx"
    model.write_bytes(b"CPU fixture only")
    return ["worker", "--out", str(tmp_path / "out"), "--onnx", str(model),
            "--syzygy-path", "CPU fixture", "--outcome-mode", worker.OUTCOME_MODE,
            "--wdl-output", "wdl", "--wdl-kind", "probabilities", "--games", "2",
            "--seed", "100", "--max-plies", "8", "--parallel-games", "2",
            "--temperature", "0", "--provider", provider, "--threads", "1",
            "--shared-target-rows", "4", "--shared-max-rows", "8",
            "--shared-batch-wait-ms", "1", "--shared-max-writes", "1",
            "--shared-unit-roster", str(roster), "--shared-unit-roster-sha256",
            digest or hashlib.sha256(roster.read_bytes()).hexdigest(),
            "--shared-max-units", "2", "--shared-max-live-games", "4",
            "--shared-deadline-seconds", "30"]


def test_cross_unit_cli_reaches_actual_finite_controller_without_new_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = fixture_module()
    handle = fixture.fake_tablebase()
    handle.close = lambda: None
    seen = []
    monkeypatch.setattr(sys, "argv", cli(tmp_path))
    monkeypatch.setattr(worker.tablebase, "open_strict_match_tablebase", lambda *_a, **_kw: handle)
    def open_session(*_args: Any, **kwargs: Any) -> tuple[Any, ...]:
        seen.append(kwargs)
        return object(), "planes", np.dtype("float32"), ("CPUExecutionProvider",), {}
    monkeypatch.setattr(worker, "open_worker_session", open_session)
    monkeypatch.setattr(worker.bt4_generation_evaluator, "BT4OnnxEvaluator", lambda *_a, **_kw: fixture.FakeEvaluator())
    original = worker.run_pooled_units
    def run(units: Any, evaluator: Any, out: Path, **options: Any) -> dict[str, Any]:
        fixture_spec = fixture.spec(tmp_path)
        units = {name: (replace(spec, initial_fen=fixture.SEVEN,
                               syzygy_path=fixture_spec.syzygy_path, model_sha256=fixture.MODEL_SHA), tb)
                 for name, (spec, tb) in units.items()}
        return original(units, evaluator, out, **options)
    monkeypatch.setattr(worker, "run_pooled_units", run)
    worker.main()
    assert len(seen) == 1
    assert seen[0]["requested_provider"] == "cpu"
    launch = json.loads((tmp_path / "out" / "cross_unit_launch.json").read_text())
    assert [unit["unit_id"] for unit in launch["units"]] == ["u0", "u1"]
    for ordinal in (7, 9):
        path = tmp_path / "out" / f"unit-{ordinal:012d}" / "raw"
        assert json.loads((path / "launch.json").read_text())["seed"] == 100 + ordinal
        assert len(list((path / "games").glob("*.npz"))) == 2


@pytest.mark.parametrize("failure", ["cuda_custody", "roster_hash", "aggregate"])
def test_cross_unit_cli_rejects_before_session_or_gpu_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    arguments = cli(tmp_path, provider="cuda" if failure == "cuda_custody" else "cpu",
                    digest="0" * 64 if failure == "roster_hash" else None)
    if failure == "aggregate":
        arguments[arguments.index("--shared-max-live-games") + 1] = "3"
    monkeypatch.setattr(sys, "argv", arguments)
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("session/canonical custody must not be touched")
    monkeypatch.setattr(worker, "open_worker_session", forbidden)
    monkeypatch.setattr(worker, "acquire_gpu_lock", forbidden)
    with pytest.raises((ValueError, SystemExit)):
        worker.main()
    assert not (tmp_path / "out").exists()


def test_measured_shared_cuda_proof_binds_each_launch_before_results(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps([{"args": {"provider": "CUDAExecutionProvider", "op_name": "Conv"}}]))
    coordinator = tmp_path / "controller"
    coordinator.mkdir()
    outputs = []
    for name in ("u0", "u1"):
        out = tmp_path / name
        out.mkdir()
        (out / "launch.json").write_text(json.dumps({"unit": name}))
        outputs.append(out)
    session = SimpleNamespace(end_profiling=lambda: str(profile),
                              get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    base = SimpleNamespace(evaluate_roots=lambda *_args: [])
    actor = worker.CudaQualifiedEvaluator(cast(worker.RootEvaluator, cast(object, base)), session, coordinator, proof_outputs=outputs)
    assert actor.evaluate_roots([], np.empty((0,))) == []
    for out in outputs:
        proof = json.loads((out / "provider_proof.json").read_text())
        assert proof["qualification_reused_same_live_session"] is True
        assert proof["shared_unit_launch_sha256"] == worker.file_sha256(out / "launch.json")
        assert proof["profile_sha256"] == worker.file_sha256(out / "provider_profile.json")
    hashes = {out: worker.file_sha256(out / "provider_proof.json") for out in outputs}
    actor.evaluate_roots([], np.empty((0,)))
    assert hashes == {out: worker.file_sha256(out / "provider_proof.json") for out in outputs}


@pytest.mark.parametrize("failure", ["launch", "profile", "provider"])
def test_actual_shared_consumer_rejects_changed_cuda_proof(tmp_path: Path, failure: str) -> None:
    fixture = fixture_module()
    spec = replace(fixture.spec(tmp_path), requested_provider="cuda", providers=("CUDAExecutionProvider",))
    spec.out.mkdir(exist_ok=True)
    (spec.out / "launch.json").write_text("launch")
    (spec.out / "provider_profile.json").write_text("profile")
    proof = {"cuda_neural_nodes": 1, "providers_after_first_call": ["CUDAExecutionProvider"],
                 "profile_sha256": worker.file_sha256(spec.out / "provider_profile.json"),
                 "shared_unit_launch_sha256": worker.file_sha256(spec.out / "launch.json")}
    if failure == "launch":
        (spec.out / "launch.json").write_text("changed")
    elif failure == "profile":
        (spec.out / "provider_profile.json").write_text("changed")
    else:
        proof["providers_after_first_call"] = ["CPUExecutionProvider"]
    (spec.out / "provider_proof.json").write_text(json.dumps(proof))
    with pytest.raises(RuntimeError, match="proof"):
        worker._evaluate_shared_roots(cast(worker.RootEvaluator, cast(object, SimpleNamespace(evaluate_roots=lambda *_: []))),
                                      [(worker.chess.Board(), np.zeros((175, 8, 8), np.float32))], [spec])


def test_controller_ack_backpressure_retains_raw_and_commits_on_owner_thread(tmp_path: Path) -> None:
    fixture, out, units, args = small_units(tmp_path)
    previous = worker.rep_fix.current()
    worker.rep_fix.apply(True, boards_discarded=True)
    owner = threading.get_ident()
    seen: dict[tuple[str, int], int] = {}
    futures: dict[tuple[str, int], tuple[Future[dict[str, Any]], dict[str, Any]]] = {}
    def acknowledge(unit: str, game: int, raw: dict[str, Any]) -> Future[dict[str, Any]]:
        assert threading.get_ident() == owner
        key = unit, game
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            raise BufferError("Ceres bounded admission")
        future: Future[dict[str, Any]] = Future()
        futures[key] = future, raw
        return future
    def control() -> None:
        assert threading.get_ident() == owner or threading.current_thread().name in ("TeacherDispatcher", "TeacherWriter")
        # SQLite ownership is exercised only by acknowledgment, never this guard.
        if threading.get_ident() == owner:
            for future, raw in futures.values():
                if not future.done():
                    future.set_result({**raw, "ceres_companion": {"fixture": True}})
    try:
        actual = worker.run_pooled_units(units, fixture.FakeEvaluator(), out, **args,
                                         owner_completed={unit: {} for unit in units},
                                         acknowledge=acknowledge, control=control)
        assert actual["status"] == "COMPLETE_NOT_ADMISSION"
        assert set(seen.values()) == {2}
        assert sum(len(values) for values in actual["unit_receipts"].values()) == 4
    finally:
        worker.rep_fix.apply(previous is True, boards_discarded=True)


def test_failed_companion_ack_raw_checkpoint_does_not_skip_owner_on_restart(tmp_path: Path) -> None:
    fixture, out, units, args = small_units(tmp_path)
    previous = worker.rep_fix.current()
    worker.rep_fix.apply(True, boards_discarded=True)
    bank: dict[tuple[str, int], dict[str, Any]] = {}
    owner = threading.get_ident()
    def lost(unit: str, game: int, raw: dict[str, Any]) -> Future[dict[str, Any]]:
        assert threading.get_ident() == owner
        bank[unit, game] = raw  # durable C3 publication happened before lost ACK
        future: Future[dict[str, Any]] = Future()
        future.set_exception(OSError("durable C3; response lost"))
        return future
    seen = []
    def recovered(unit: str, game: int, raw: dict[str, Any]) -> dict[str, Any]:
        assert threading.get_ident() == owner
        if (unit, game) in bank:
            assert bank[unit, game] == raw
        seen.append((unit, game))
        return {**raw, "ceres_companion": {"recovered": True}}
    try:
        with pytest.raises(OSError, match="response lost"):
            worker.run_pooled_units(units, fixture.FakeEvaluator(), out, **args,
                                     owner_completed={unit: {} for unit in units}, acknowledge=lost)
        assert bank
        result = worker.run_pooled_units(units, fixture.FakeEvaluator(), out, resume=True, **args,
                                         owner_completed={unit: {} for unit in units}, acknowledge=recovered)
        assert len(seen) == 4
        assert all(receipt["ceres_companion"]["recovered"]
                   for receipts in result["unit_receipts"].values() for receipt in receipts)
        assert sum(len(list(spec.out.glob("games/*.npz"))) for spec, _ in units.values()) == 4
    finally:
        worker.rep_fix.apply(previous is True, boards_discarded=True)
