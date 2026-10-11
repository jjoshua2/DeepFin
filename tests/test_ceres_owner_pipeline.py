"""CPU socket RPC through retained history verification and durable publication."""
from __future__ import annotations

from dataclasses import replace
from concurrent.futures import Future
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import chess
import pytest

from scripts import bt4_root_policy_worker as worker
from scripts.ceres_pipelined_client import PipelinedCompanion
from scripts.ceres_shared_game_service import bind_ceres_raw_backend, bind_retained_ceres_history, serve_ceres_stream
from scripts.shared_teacher_owner import bind_owner_checkpoints
from scripts import ceres_shared_game_service as shared_service
from tests.test_cross_unit_teacher_generation import fixture_module
from tests.teacher_reference_fixture import reference_sources, cpu_config


def retained_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, Any, Any, Any]:
    reference = reference_sources(tmp_path)
    root = reference / "scripts"
    def load(name: str, path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module
    publisher = load("ceres_atomic_publication_v1", root / "ceres_atomic_publication_v1.py")
    retained = load("retained_ceres_service", root / "ceres_dynamic_game_service.py")
    comparison = load("retained_comparison", reference / "reference/compare.py")
    comparison.BT = comparison.C3 = Path(__file__).resolve().parents[1]
    comparison.FROZEN = reference / "encoding/ceres_tpg.py"
    comparison.AD = reference / "ad"
    # Imports the exact native history encoder/oracle, never opens an ORT session.
    before = list(sys.path)
    try:
        api = comparison.imports("bt4")
    finally:
        sys.path[:] = before
    return retained, comparison, publisher, api


@pytest.mark.parametrize("failure", ["visibility", "deadline"])
def test_child_setup_rejects_before_retained_imports_or_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"shared_service": {"units": [], "max_games": 1, "target_rows": 1,
        "max_rows": 1, "batch_wait_ms": 1, "poll_seconds": 0.001, "deadline_seconds": 1, "physical_batch": 32,
        "source_pins": {}, "retained_service_source": {}, "publisher_source": {}}}))
    monkeypatch.setattr(sys, "argv", ["service", "--config", str(config), "--fake-cpu"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0" if failure == "visibility" else "-1")
    monkeypatch.setenv("DEEPFIN_COMPARE_DEADLINE", str(time.monotonic() - 1))
    with pytest.raises((ValueError, TimeoutError), match="before setup"):
        shared_service.main()
    assert list(tmp_path.iterdir()) == [config]


@pytest.mark.parametrize("failure", ["real_mode", "encoder_pin"])
def test_cpu_reference_relocation_rejects_real_mode_or_unreviewed_encoder_before_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    reference = reference_sources(tmp_path)
    root = Path(__file__).resolve().parents[1]
    config = cpu_config(reference, root)
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    config["shared_service"] = {"units": [], "max_games": 1, "target_rows": 1, "max_rows": 1,
        "batch_wait_ms": 1, "poll_seconds": 0.001, "deadline_seconds": 10, "physical_batch": 32,
        "source_pins": {relative: digest(root / relative) for relative in (
            "chess_anti_engine/teacher_dispatch.py", "scripts/ceres_shared_game_service.py",
            "scripts/ceres_raw_backend.py", "scripts/shared_teacher_generation.py")},
        "retained_service_source": {"path": str(reference / "scripts/ceres_dynamic_game_service.py"),
            "sha256": digest(reference / "scripts/ceres_dynamic_game_service.py")},
        "publisher_source": {"path": str(reference / "scripts/ceres_atomic_publication_v1.py"),
            "sha256": digest(reference / "scripts/ceres_atomic_publication_v1.py")}}
    if failure == "encoder_pin":
        config["qualified_plan"]["source_pins"][config["qualified_plan"]["cpu_fixture_paths"]["tpg"]] = "0" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    argv = ["service", "--config", str(path)]
    if failure != "real_mode":
        argv += ["--fake-cpu"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.delenv("DEEPFIN_COMPARE_DEADLINE", raising=False)
    with pytest.raises(ValueError, match=r"CPU reference relocation|exact retained CPU encoder"):
        shared_service.main()


def test_actual_pipelined_local_ids_retained_history_three_heads_and_orphan_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, comparison, publisher, api = retained_modules(monkeypatch, tmp_path)
    fixture = fixture_module()
    worker.rep_fix.apply(True, boards_discarded=True)
    units = {}
    raw_receipts = {}
    moves = ("f2f3", "e7e5", "g2g4", "d8h4")
    class MateEvaluator:
        def evaluate_roots(self, boards: Any, x_batch: np.ndarray) -> list[Any]:
            return [fixture.FakeEvaluator(moves[board.ply()]).evaluate_roots([board], x[None])[0]
                    for board, x in zip(boards, x_batch)]
    for unit in ("u0", "u1"):
        spec = replace(fixture.spec(tmp_path), out=tmp_path / unit, games=1, parallel_games=1,
                       initial_fen=chess.STARTING_FEN, temperature=1, max_plies=8)
        result = worker.run_worker(spec, MateEvaluator(), fixture.fake_tablebase())
        units[unit] = spec.out.resolve()
        receipt = result["game_files"][0]
        raw_receipts[unit] = {**receipt, "path": str(units[unit] / "games" / receipt["path"])}
    load, publish = bind_retained_ceres_history(retained=retained, comparison=comparison, publisher=publisher,
                                               api=api, units=units, model=comparison.MODELS["ceres"],
                                               fake_cpu=True, physical_batch=256, provider=None)
    calls = []
    class Session:
        def run(self, names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            assert names == ["policy", "value", "value2"]
            assert feeds["squares_byte"].shape == (256, 64, 137)
            calls.append(feeds["squares_byte"].copy())
            return [np.full((256, 1858), 2, np.float16),
                    np.tile(np.array([3, 4, 5], np.float16), (256, 1)),
                    np.tile(np.array([-6, -7, -8], np.float16), (256, 1))]
    counts: dict[str, int] = {}
    infer = bind_ceres_raw_backend(session=Session(), physical_batch=256,
                                   gather_context=api[-2].ceres_tpg_gather_context,
                                   gather_indices=api[6], accounting=counts)
    def exchange() -> list[dict[str, Any]]:
        client_socket, server_socket = socket.socketpair()
        client_input, client_output = client_socket.makefile("wb"), client_socket.makefile("rb")
        incoming, outgoing = server_socket.makefile("r"), server_socket.makefile("w")
        failures = []
        def serve() -> None:
            try:
                serve_ceres_stream(incoming, outgoing, total_games=256, max_games=2, target_rows=8,
                                   max_rows=8, batch_wait_ms=100, poll_seconds=0.001, deadline_seconds=10,
                                   load_roots=load, infer=infer, publish=publish, unit_roster=("u0", "u1"))
            except BaseException as exc:
                failures.append(exc)
        thread = threading.Thread(target=serve)
        thread.start()
        # Consume the retained startup-ready handshake before adopting its child.
        assert json.loads(client_output.readline())["state"] == "READY_FOR_GAME"
        child = SimpleNamespace(stdin=client_input, stdout=client_output, pid=1, poll=lambda: None)
        client = SimpleNamespace(child=child, guard=lambda: None, check_priority=lambda: None,
                                 charge=lambda _count: None, seen=[],
                                 cleanup=SimpleNamespace(members=lambda _pid: [], descendants=lambda _pid: []))
        pipeline = PipelinedCompanion(client, ("u0", "u1"), max_pending=2, wire_bytes=8192)
        futures = [pipeline.submit(unit, raw_receipts[unit], 0) for unit in ("u0", "u1")]
        assert set(pipeline.pending) == {0, 128}
        deadline = time.monotonic() + 10
        try:
            while not all(future.done() for future in futures):
                pipeline.poll()
                assert time.monotonic() < deadline
                time.sleep(0.001)
            values = [future.result() for future in futures]
            client_socket.sendall(b'{"op":"stop"}\n')
            thread.join(10)
            assert not thread.is_alive()
            assert not failures
            return values
        finally:
            client_input.close()
            client_output.close()
            incoming.close()
            outgoing.close()
            client_socket.close()
            server_socket.close()
    results = exchange()
    assert len(calls) == 1
    assert counts == {"calls": 1, "real_rows": 8, "padding_rows": 248, "physical_rows": 256}
    for unit, result in zip(units, results):
        companion = Path(result["ceres_companion"]["path"])
        receipt = json.loads(companion.read_text())
        assert receipt["unit_id"] == unit
        assert receipt["game_id"] == 0
        assert receipt["root_ids"] == [[0, index] for index in range(4)]
        assert receipt["physical_rows"] is None
        with np.load(receipt["labels"]["path"], allow_pickle=False) as arrays:
            assert np.array_equal(arrays["policy_logits"], np.full((4, 1858), 2, np.float16))
            assert arrays["value_logits"].tolist() == [[3, 4, 5]] * 4
            assert arrays["value2_logits"].tolist() == [[-6, -7, -8]] * 4
        companion.unlink()  # controlled CPU fixture: simulate label-before-JSON death
    recovered = exchange()
    assert len(calls) == 1  # embedded durable metadata restores JSON without reinference
    assert results == recovered


def test_pipeline_backpressure_and_duplicate_fail_closed(tmp_path: Path) -> None:
    left, right = socket.socketpair()
    stream_in, stream_out = left.makefile("wb"), left.makefile("rb")
    child = SimpleNamespace(stdin=stream_in, stdout=stream_out, pid=1, poll=lambda: None)
    client = SimpleNamespace(child=child, guard=lambda: None, check_priority=lambda: None, charge=lambda _: None,
                             seen=[], cleanup=SimpleNamespace(members=lambda _: [], descendants=lambda _: []))
    try:
        pipeline = PipelinedCompanion(client, ("u0", "u1"), max_pending=1, wire_bytes=8192)
        raw = {"path": str(tmp_path / "game_00000000.npz"), "sha256": hashlib.sha256(b"fixture").hexdigest(), "rows": 0}
        future = pipeline.submit("u0", raw, 0)
        with pytest.raises(BufferError):
            pipeline.submit("u1", raw, 0)
        with pytest.raises(ValueError, match="duplicate"):
            pipeline.submit("u0", raw, 0)
        right.sendall(b'{"request_id":128,"state":"GAME_DURABLE"}\n')
        with pytest.raises(ValueError, match="unknown"):
            pipeline.poll()
        with pytest.raises(ValueError, match="unknown"):
            future.result()
        assert 0 in pipeline.pending
        assert 0 in pipeline.submitted
    finally:
        stream_in.close()
        stream_out.close()
        left.close()
        right.close()


def test_retained_sqlite_ack_waits_for_ceres_and_preserves_completed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, _comparison, _publisher, _api = retained_modules(monkeypatch, tmp_path)
    reference = reference_sources(tmp_path)
    checkpoint_module = retained.load("owner_checkpoint_fixture", str(reference / "scripts/bt4_longblock_checkpoint_v2.py"))
    recovery = retained.load("owner_recovery_fixture", str(reference / "scripts/bt4_raw_recovery_v2.py"))
    fixture = fixture_module()
    worker.rep_fix.apply(True, boards_discarded=True)
    owner = threading.get_ident()
    def guard() -> None:
        assert threading.get_ident() == owner
    limits = {"max_attempts": 2, "unit_seconds": 1, "unit_io_bytes": 1, "unit_output_bytes": 1,
                  "total_seconds": 2, "total_io_bytes": 2, "total_output_bytes": 2}
    checkpoint = checkpoint_module.Checkpoint(tmp_path / "checkpoint.sqlite", {"fixture": "CPU"}, limits, guard)
    root = tmp_path / "raw"
    attempt = checkpoint.begin(7, {"pid": 1, "start_ticks": 1, "boot_id": "CPU fixture"}, str(root))
    assert not root.exists()  # the retained begin() freshness boundary is real
    spec = replace(fixture.spec(tmp_path), out=root, games=2, parallel_games=1)
    game_files = worker.run_worker(spec, fixture.FakeEvaluator(), fixture.fake_tablebase(),
                                  shared_dispatch=(1, 2, 1, 1))["game_files"]
    raw = game_files[0]
    requests = []
    def submit(unit: str, verified: dict[str, Any], game: int) -> Future[dict[str, Any]]:
        assert (unit, game) == ("u0", 0)
        future: Future[dict[str, Any]] = Future()
        requests.append((future, verified))
        return future
    def verify_raw(_unit: str, path: Path, game: int) -> dict[str, Any]:
        return recovery.verified_game(path, game, checkpoint.binding, fixture.MODEL_SHA, guard, lambda _: None)
    def verify_completed(_unit: str, receipt: dict[str, Any]) -> None:
        assert receipt["ceres_companion"] == {"fixture": "durable C3"}
    try:
        completed, acknowledge = bind_owner_checkpoints(
            units={"u0": (7, root, checkpoint, attempt)}, verify_raw=verify_raw,
            verify_completed=verify_completed, companion=SimpleNamespace(submit=submit))
        assert completed == {"u0": {}}
        pending = acknowledge("u0", 0, raw)
        assert isinstance(pending, Future)
        assert not pending.done()
        assert checkpoint.completed(7, lambda _: None) == ()
        requests[0][0].set_result({**requests[0][1], "ceres_companion": {"fixture": "durable C3"}})
        assert pending.result()["ceres_companion"] == {"fixture": "durable C3"}
        assert checkpoint.completed(7, lambda _: None) == (0,)
        completed, _ack = bind_owner_checkpoints(
            units={"u0": (7, root, checkpoint, attempt)}, verify_raw=verify_raw,
            verify_completed=verify_completed, companion=SimpleNamespace(submit=submit))
        assert completed["u0"][0] == pending.result()
        saved = checkpoint.db.execute("SELECT receipt FROM games WHERE unit=7 AND game=0").fetchone()[0]
        for key, value in (("sha256", "0" * 64), ("rows", 2), ("status", "discarded"),
                           ("launch_sha256", "0" * 64), ("discarded_rows", 9),
                           ("path", "game_00000001.npz")):
            corrupt = json.loads(saved)
            corrupt["canonical_raw"][key] = value
            checkpoint.db.execute("UPDATE games SET receipt=? WHERE unit=7 AND game=0", (json.dumps(corrupt),))
            with pytest.raises(ValueError, match="canonical"):
                bind_owner_checkpoints(units={"u0": (7, root, checkpoint, attempt)}, verify_raw=verify_raw,
                                       verify_completed=verify_completed, companion=SimpleNamespace(submit=submit))
        corrupt = json.loads(saved)
        corrupt["path"] = str(tmp_path / "foreign_unit" / "games" / raw["path"])
        checkpoint.db.execute("UPDATE games SET receipt=? WHERE unit=7 AND game=0", (json.dumps(corrupt),))
        with pytest.raises(ValueError, match="unit attempt roots"):
            bind_owner_checkpoints(units={"u0": (7, root, checkpoint, attempt)}, verify_raw=verify_raw,
                                   verify_completed=verify_completed, companion=SimpleNamespace(submit=submit))
        game_one = verify_raw("u0", root / "games" / game_files[1]["path"], 1)
        corrupt = {**game_one, "canonical_raw": game_files[1], "ceres_companion": {"fixture": "durable C3"}}
        checkpoint.db.execute("UPDATE games SET receipt=? WHERE unit=7 AND game=0", (json.dumps(corrupt),))
        with pytest.raises(ValueError, match="game IDs disagree"):
            bind_owner_checkpoints(units={"u0": (7, root, checkpoint, attempt)}, verify_raw=verify_raw,
                                   verify_completed=verify_completed, companion=SimpleNamespace(submit=submit))
        checkpoint.db.execute("UPDATE games SET receipt=? WHERE unit=7 AND game=0", (saved,))
        with pytest.raises(ValueError, match="active unit"):
            bind_owner_checkpoints(units={"u0": (8, root, checkpoint, attempt)}, verify_raw=verify_raw,
                                   verify_completed=verify_completed, companion=SimpleNamespace(submit=submit))
        with pytest.raises(ValueError, match="full unit"):
            checkpoint.finish(attempt, True)  # one CPU game earns no128-game completion
        checkpoint.finish(attempt, False)
    finally:
        checkpoint.close()


@pytest.mark.parametrize("physical", [32, 256, 512])
def test_retained_companion_starts_actual_fake_child_and_pipelines_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, physical: int,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    retained, comparison, _publisher, _api = retained_modules(monkeypatch, tmp_path)
    reference = reference_sources(tmp_path)
    client_module = retained.load("retained_child_client", str(reference / "scripts/ceres_game_companion_client.py"))
    fixture = fixture_module()
    worker.rep_fix.apply(True, boards_discarded=True)
    units, receipts = {}, {}
    for name in ("u0", "u1"):
        spec = replace(fixture.spec(tmp_path), out=tmp_path / name, games=1, parallel_games=1)
        raw = worker.run_worker(spec, fixture.FakeEvaluator(), fixture.fake_tablebase())["game_files"][0]
        units[name] = spec.out.resolve()
        receipts[name] = {**raw, "path": str(spec.out.resolve() / "games" / raw["path"])}
    root = Path(__file__).resolve().parents[1]
    config = cpu_config(reference, root)
    service = root / "scripts/ceres_shared_game_service.py"
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    config.update(service_source={"path": str(service), "sha256": digest(service)},
                  service_output=str(tmp_path / "ceres_service"))
    config["shared_service"] = {"units": [{"unit_id": name, "raw_root": str(path)} for name, path in units.items()],
        "max_games": 2, "target_rows": 2, "max_rows": 8, "batch_wait_ms": 100, "poll_seconds": 0.001,
        "deadline_seconds": 10, "physical_batch": physical,
        "source_pins": {relative: digest(root / relative) for relative in (
            "chess_anti_engine/teacher_dispatch.py", "scripts/ceres_shared_game_service.py",
            "scripts/ceres_raw_backend.py", "scripts/shared_teacher_generation.py")},
        "retained_service_source": {"path": retained.__file__, "sha256": digest(Path(retained.__file__))},
        "publisher_source": {"path": _publisher.__file__, "sha256": digest(Path(_publisher.__file__))}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    deadline = time.monotonic() + 20
    def guard() -> None:
        assert time.monotonic() < deadline
    client = client_module.Companion(config_path, -1, guard, lambda _: None, fake_cpu=True)
    child = client.child
    pipeline = PipelinedCompanion(client, tuple(units), max_pending=2, wire_bytes=8192)
    try:
        futures = [pipeline.submit(name, receipts[name], 0) for name in units]
        while not all(future.done() for future in futures):
            pipeline.poll()
            time.sleep(0.001)
        for name, future in zip(units, futures):
            receipt = json.loads(Path(future.result()["ceres_companion"]["path"]).read_text())
            assert receipt["unit_id"] == name
            assert receipt["fake_cpu"] is True
            assert receipt["model"] == comparison.MODELS["ceres"]
            with np.load(receipt["labels"]["path"], allow_pickle=False) as arrays:
                assert arrays["value_logits"].tolist() == [[1, 2, 3]]
                assert arrays["value2_logits"].tolist() == [[4, 5, 6]]
    finally:
        client.close()  # exact retained owned-process cleanup, no new reaper
    assert child.poll() == 0
    result = json.loads((tmp_path / "ceres_service/RESULT.json").read_text())
    assert result["status"] == "CPU_FAKE_ONLY"
    assert result["successful_validated_physical_accounting"] == {"calls": 1, "real_rows": 2,
        "physical_rows": physical, "padding_rows": physical - 2}
