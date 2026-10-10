"""CPU-only protocol and canonical board/history/raw publication integration."""
from __future__ import annotations

import importlib.util
import io
from collections.abc import Sequence
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.teacher_dispatch import TeacherDispatcher
from scripts import bt4_root_policy_worker as worker
from scripts.ceres_shared_game_service import checked_ceres_backend, serve_ceres_stream
from scripts.shared_teacher_generation import CompletedGameLabels, DurableWriter


@pytest.fixture(autouse=True)
def actual_history_mode() -> Any:
    previous = worker.rep_fix.current()
    worker.rep_fix.apply(True, boards_discarded=True)
    yield
    worker.rep_fix.apply(previous is True, boards_discarded=True)


def fixture_module() -> Any:
    spec = importlib.util.spec_from_file_location("worker_fixture", Path(__file__).with_name("test_bt4_root_policy_worker.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def poll(operation: Any, *, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while operation():
        if time.monotonic() >= deadline:
            raise AssertionError("CPU fixture did not finish within finite budget")
        time.sleep(0.001)


def test_cross_game_full_tail_and_durable_ack_retry() -> None:
    calls = []
    def evaluate(rows: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        calls.append(tuple(rows))
        return list(rows)
    dispatcher = TeacherDispatcher(evaluate, target_rows=512, max_rows=1024, batch_wait_ms=20)
    bank: dict[int, tuple[Any, ...]] = {}
    fail = [True]
    def publish(game_id: int, _sha: str, values: tuple[Any, ...]) -> None:
        expected = tuple((game_id, i) for i in range(300))
        assert values == expected
        if game_id in bank:
            assert bank[game_id] == values
        else:
            bank[game_id] = values
        if fail[0]:
            fail[0] = False
            raise OSError("publication completed; acknowledgment lost")
    service = CompletedGameLabels(dispatcher, max_games=2, total_games=2, max_game_rows=400, publish=publish)
    try:
        service.submit(0, "a" * 64, [(0, i) for i in range(300)])
        service.submit(1, "b" * 64, [(1, i) for i in range(300)])
        assert service.drain(stopped=True) == []
        deadline = time.monotonic() + 5
        observed_failure = False
        while service.pending:
            try:
                service.drain()
            except OSError:
                observed_failure = True
                assert 0 in service.pending
                assert 0 not in service.durable
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert observed_failure
        assert sorted(bank) == [0, 1]
        assert dispatcher.histogram == {512: 1, 88: 1}
        assert calls[0][-1] == (1, 211)
        assert calls[1][0] == (1, 212)
        with pytest.raises(ValueError, match="distinct"):
            service.submit(0, "a" * 64, [])
        with pytest.raises(ValueError, match="distinct"):
            service.submit(2, "a" * 64, [])
    finally:
        dispatcher.close(timeout=5)
        service.close(timeout=5)


def test_tail_deadline_duplicate_backpressure_and_shutdown_budget() -> None:
    entered, release = threading.Event(), threading.Event()
    def evaluate(rows: Sequence[int]) -> list[int]:
        entered.set()
        assert release.wait(5)
        return list(rows)
    dispatcher = TeacherDispatcher(evaluate, target_rows=512, max_rows=512, batch_wait_ms=1)
    result = dispatcher.submit("one", [1])
    try:
        assert entered.wait(5)  # low-volume tail dispatched without end-of-input
        assert result.cancel() is False
        with pytest.raises(ValueError, match="duplicate"):
            dispatcher.submit("one", [1])
        with pytest.raises(BufferError):
            dispatcher.submit("overflow", list(range(512)))
        with pytest.raises(ValueError, match="finite"):
            dispatcher.close(timeout=float("inf"))
        with pytest.raises(TimeoutError, match="ownership"):
            dispatcher.close(timeout=0.001)
        release.set()
        assert result.result(timeout=5) == (1,)
    finally:
        release.set()
        dispatcher.close(timeout=5)


def test_backend_failure_poison_reaches_every_owned_request() -> None:
    dispatcher = TeacherDispatcher(lambda rows: [], target_rows=4, max_rows=8, batch_wait_ms=20)
    first = dispatcher.submit("first", [1, 2])
    second = dispatcher.submit("second", [3, 4])
    try:
        for future in (first, second):
            with pytest.raises(ValueError, match="output count"):
                future.result(timeout=5)
        with pytest.raises(RuntimeError, match="failed"):
            dispatcher.submit("third", [5])
    finally:
        dispatcher.close(timeout=5)


def test_writer_backpressure_and_finite_hung_writer_shutdown() -> None:
    entered, release = threading.Event(), threading.Event()
    writer = DurableWriter(capacity=1)
    def blocked() -> str:
        entered.set()
        assert release.wait(5)
        return "durable"
    future = writer.submit(blocked)
    try:
        assert entered.wait(5)
        with pytest.raises(BufferError):
            writer.submit(lambda: None)
        with pytest.raises(TimeoutError, match="still owned"):
            writer.close(timeout=0.001)
        release.set()
        assert future.result(timeout=5) == "durable"
    finally:
        release.set()
        writer.close(timeout=5)


@pytest.mark.parametrize("capacity", [1, 2])
def test_real_worker_shared_callsite_preserves_history_targets_and_raw_npz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capacity: int,
) -> None:
    fixture = fixture_module()
    monkeypatch.setattr(worker.rep_fix, "current", lambda: True)
    control_dir = tmp_path / "control"
    candidate_dir = tmp_path / "candidate"
    control_dir.mkdir()
    candidate_dir.mkdir()
    control = replace(fixture.spec(control_dir), games=6, parallel_games=capacity)
    candidate = replace(fixture.spec(candidate_dir), games=6, parallel_games=capacity,
                        syzygy_path=control.syzygy_path)
    expected = worker.run_worker(control, fixture.FakeEvaluator(), fixture.fake_tablebase())
    actual = worker.run_worker(candidate, fixture.FakeEvaluator(), fixture.fake_tablebase(), shared_dispatch=(2, 4, 1, 1))
    assert actual["games"] == 6
    assert actual["rows_emitted"] == expected["rows_emitted"] == 6
    for game_id in range(6):
        name = f"game_{game_id:08d}.npz"
        expected_meta, expected_arrays = fixture.read_game(control.out / "games" / name)
        actual_meta, actual_arrays = fixture.read_game(candidate.out / "games" / name)
        assert expected_meta == actual_meta
        assert actual_meta["rows"][0]["teacher"]["history_rep_fix"] is True
        for key in expected_arrays:
            assert expected_arrays[key].dtype == actual_arrays[key].dtype
            assert np.array_equal(expected_arrays[key], actual_arrays[key])
        assert (candidate.out / "games" / f"game_{game_id:08d}.checkpoint.json").exists()


def test_actual_multi_ply_history_and_persisted_resume_contract(tmp_path: Path) -> None:
    import chess
    fixture = fixture_module()
    directory = tmp_path / "multi"
    directory.mkdir()
    spec = replace(fixture.spec(directory), games=6, parallel_games=2,
                   initial_fen=chess.STARTING_FEN, temperature=1.0, max_plies=8)
    moves = ("f2f3", "e7e5", "g2g4", "d8h4")
    class MateEvaluator:
        def evaluate_roots(self, boards: list[chess.Board], x_batch: np.ndarray) -> list[Any]:
            return [fixture.FakeEvaluator(moves[board.ply()]).evaluate_roots([board], row[None])[0]
                    for board, row in zip(boards, x_batch)]
    evaluator = MateEvaluator()
    control_spec = replace(spec, out=tmp_path / "control-multi", parallel_games=1)
    worker.run_worker(control_spec, evaluator, fixture.fake_tablebase())
    summary = worker.run_worker(spec, evaluator, fixture.fake_tablebase(), shared_dispatch=(2, 4, 1, 1))
    assert summary["rows_emitted"] == 24
    for receipt in summary["game_files"]:
        metadata, arrays = fixture.read_game(spec.out / "games" / receipt["path"])
        control_meta, control_arrays = fixture.read_game(control_spec.out / "games" / receipt["path"])
        assert metadata == control_meta
        assert all(np.array_equal(arrays[key], control_arrays[key]) for key in arrays)
        assert [row["move_uci"] for row in metadata["rows"]] == list(moves)
        assert [row["ply_index"] for row in metadata["rows"]] == [0, 1, 2, 3]
        assert arrays["x"].shape == (4, 175, 8, 8)
        assert not np.array_equal(arrays["x"][0], arrays["x"][3])
    args: dict[str, Any] = {"target_rows": 2, "max_rows": 4, "batch_wait_ms": 1, "max_writes": 1, "resume": True}
    resumed = worker.run_pooled_games(spec, evaluator, fixture.fake_tablebase(), spec.out / "games", **args)
    assert resumed["histogram"] == {}
    assert len(resumed["receipts"]) == 6
    for changed in (replace(spec, seed=15), replace(spec, model_sha256="b" * 64),
                    replace(spec, max_plies=7), replace(spec, temperature=0.5)):
        with pytest.raises(ValueError, match="resume"):
            worker.run_pooled_games(changed, evaluator, fixture.fake_tablebase(), spec.out / "games", **args)
    # Recover a raw orphan by replaying its exact initial board/seed, with no second file.
    (spec.out / "games" / "game_00000003.checkpoint.json").unlink()
    resumed = worker.run_pooled_games(spec, evaluator, fixture.fake_tablebase(), spec.out / "games", **args)
    assert len(resumed["receipts"]) == 6
    assert sum(size * calls for size, calls in resumed["histogram"].items()) == 4
    assert len(list((spec.out / "games").glob("*.npz"))) == 6


@pytest.mark.parametrize("failure", ["before_raw", "after_raw"])
def test_worker_durable_failure_and_orphan_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    fixture = fixture_module()
    spec = fixture.spec(tmp_path)
    with monkeypatch.context() as patch:
        if failure == "before_raw":
            def fail_raw(*_args: Any, **_kwargs: Any) -> Any:
                raise OSError("before raw publication")
            patch.setattr(worker, "write_finalized_game", fail_raw)
        else:
            original_json = worker._atomic_json
            def fail_checkpoint(path: Path, document: Any) -> None:
                if path.name.endswith("checkpoint.json"):
                    raise OSError("after raw publication before checkpoint ack")
                original_json(path, document)
            patch.setattr(worker, "_atomic_json", fail_checkpoint)
        with pytest.raises(OSError, match="publication"):
            worker.run_worker(spec, fixture.FakeEvaluator(), fixture.fake_tablebase(), shared_dispatch=(2, 4, 1, 1))
    assert not (spec.out / "summary.json").exists()
    recovered = worker.run_pooled_games(spec, fixture.FakeEvaluator(), fixture.fake_tablebase(), spec.out / "games",
                                       target_rows=2, max_rows=4, batch_wait_ms=1, max_writes=1, resume=True)
    assert len(recovered["receipts"]) == 2
    assert len(list((spec.out / "games").glob("*.npz"))) == 2
    assert len(list((spec.out / "games").glob("*.checkpoint.json"))) == 2


def test_actual_worker_stop_with_admitted_results_and_restart(tmp_path: Path) -> None:
    fixture = fixture_module()
    spec = fixture.spec(tmp_path)
    class StopEvaluator(fixture.FakeEvaluator):
        def evaluate_roots(self, boards: Any, x_batch: Any) -> Any:
            result = super().evaluate_roots(boards, x_batch)
            (spec.out / "STOP").write_text("intentional CPU fixture stop\n")
            return result
    paused = worker.run_worker(spec, StopEvaluator(), fixture.fake_tablebase(), shared_dispatch=(2, 4, 1, 1))
    assert paused["status"] == "paused"
    assert not (spec.out / "summary.json").exists()
    assert not list((spec.out / "games").glob("*.npz"))
    (spec.out / "STOP").unlink()
    resumed = worker.run_pooled_games(spec, fixture.FakeEvaluator(), fixture.fake_tablebase(), spec.out / "games",
                                     target_rows=2, max_rows=4, batch_wait_ms=1, max_writes=1, resume=True)
    assert len(resumed["receipts"]) == 2


def test_pipelined_ceres_wire_full_tail_and_three_raw_heads(tmp_path: Path) -> None:
    server, client = socket.socketpair()
    client.settimeout(5)
    calls = []
    result: list[Any] = []
    def load_roots(message: dict[str, Any]) -> tuple[Any, ...]:
        game = message["game_id"]
        roots = []
        for index in range(300):
            feed = np.zeros((64, 137), dtype=np.uint8)
            feed.flat[:2] = [index % 256, index // 256]
            roots.append(SimpleNamespace(slot_id=game * 1024 + index, fen=f"game{game}-root{index}", feed=feed))
        return tuple(roots)
    def infer(roots: Sequence[Any]) -> list[Any]:
        calls.append(len(roots))
        return [SimpleNamespace(slot_id=r.slot_id, fen=r.fen, feed=r.feed.copy(),
                                feed_sha256=hashlib.sha256(r.feed.tobytes()).hexdigest(),
                                policy_logits=np.full(1858, r.slot_id, np.float16),
                                value_logits=np.full(3, 10, np.float16),
                                value2_logits=np.full(3, 20, np.float16)) for r in roots]
    def publish(message: dict[str, Any], values: tuple[Any, ...]) -> dict[str, Any]:
        game = message["game_id"]
        path = tmp_path / f"labels-{game}.npz"
        with path.open("xb") as handle:
            np.savez(handle, root_id=np.array([[game, i] for i in range(300)]),
                     raw_sha=np.array(message["raw_sha256"]),
                     policy_logits=np.stack([value.policy_logits for value in values]),
                     value_logits=np.stack([value.value_logits for value in values]),
                     value2_logits=np.stack([value.value2_logits for value in values]))
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(path, allow_pickle=False) as archive:
            assert archive["root_id"].tolist() == [[game, i] for i in range(300)]
            assert archive["raw_sha"].item() == message["raw_sha256"]
            assert archive["policy_logits"].dtype == np.float16
            assert archive["value_logits"][0].tolist() == [10, 10, 10]
            assert archive["value2_logits"][0].tolist() == [20, 20, 20]
        return {"companion": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    def run() -> None:
        try:
            with server.makefile("r") as incoming, server.makefile("w") as outgoing:
                result.append(serve_ceres_stream(incoming, outgoing, total_games=2, max_games=2,
                                                target_rows=512, max_rows=1024, batch_wait_ms=100,
                                                poll_seconds=0.001, deadline_seconds=5,
                                                load_roots=load_roots, infer=infer, publish=publish))
        except BaseException as exc:
            result.append(exc)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        # Both requests are on the wire before any response: actual pipelined RPC.
        client.sendall(b'{"op":"game","game_id":0,"raw_sha256":"' + b'a' * 64 + b'"}\n'
                       + b'{"op":"game","game_id":1,"raw_sha256":"' + b'b' * 64 + b'"}\n'
                       + b'{"op":"stop"}\n')
        with client.makefile("r") as replies:
            assert json.loads(replies.readline())["state"] == "READY_FOR_GAME"
            first = json.loads(replies.readline())
            second = json.loads(replies.readline())
            assert {first["game_id"], second["game_id"]} == {0, 1}
            assert first["state"] == second["state"] == "GAME_DURABLE"
            assert json.loads(replies.readline())["games"] == 2
        thread.join(5)
        assert not thread.is_alive()
        assert calls == [512, 88]
        assert result[0]["logical_batch_histogram"] == {512: 1, 88: 1}
    finally:
        client.close()
        server.close()


def test_ceres_stale_identity_and_oversized_request_fail_loud() -> None:
    root = SimpleNamespace(slot_id=1, fen="exact", feed=np.zeros((64, 137), np.uint8))
    backend = checked_ceres_backend(lambda _roots: [SimpleNamespace(slot_id=2, fen="stale", feed=root.feed,
                                      feed_sha256=hashlib.sha256(root.feed.tobytes()).hexdigest())])
    with pytest.raises(ValueError, match="misrouted"):
        backend([root])
    dispatcher = TeacherDispatcher(lambda rows: rows, target_rows=4, max_rows=4, batch_wait_ms=0)
    try:
        with pytest.raises(ValueError, match="single"):
            dispatcher.submit("impossible", [0] * 5)
    finally:
        dispatcher.close(timeout=5)


def test_flush_drains_every_admitted_chunk_and_telemetry_stays_bounded() -> None:
    dispatcher = TeacherDispatcher(lambda rows: rows, target_rows=2, max_rows=8, batch_wait_ms=30000)
    try:
        result = dispatcher.submit("tail", [0, 1, 2, 3, 4])
        dispatcher.flush()
        assert result.result(timeout=1) == (0, 1, 2, 3, 4)
        assert dispatcher.histogram == {2: 2, 1: 1}
        for i in range(140):
            result = dispatcher.submit(f"sample-{i}", [i])
            dispatcher.flush()
            assert result.result(timeout=1) == (i,)
        assert len(dispatcher.queue_wait_seconds) == 128
        assert dispatcher.histogram == {2: 2, 1: 141}
        assert len(dispatcher.histogram) <= dispatcher.target_rows
    finally:
        dispatcher.close(timeout=5)


@pytest.mark.parametrize("feed", [np.zeros((64, 137), np.float64), np.zeros((137, 64), np.uint8)])
def test_ceres_output_feed_requires_exact_uint8_geometry(feed: np.ndarray) -> None:
    root = SimpleNamespace(slot_id=1, fen="exact", feed=np.zeros((64, 137), np.uint8))
    value = SimpleNamespace(slot_id=1, fen="exact", feed=feed,
                            feed_sha256=hashlib.sha256(root.feed.tobytes()).hexdigest())
    with pytest.raises(ValueError, match="feed identity"):
        checked_ceres_backend(lambda _roots: [value])([root])


def test_hard_death_after_stage_fsync_recovers_exact_file(tmp_path: Path) -> None:
    fixture = fixture_module()
    spec = fixture.spec(tmp_path)
    child = os.fork()
    if child == 0:
        # Real process death skips the canonical writer's finally cleanup.
        def die_before_rename(_source: Any, _target: Any) -> None:
            os._exit(73)
        original = worker.os.replace
        def only_raw_stage(source: Any, target: Any) -> None:
            if str(source).endswith(".npz.writing"):
                die_before_rename(source, target)
            original(source, target)
        worker.os.replace = only_raw_stage
        worker.run_worker(spec, fixture.FakeEvaluator(), fixture.fake_tablebase(), shared_dispatch=(2, 4, 1, 1))
        os._exit(74)
    deadline = time.monotonic() + 10
    while True:
        finished, status = os.waitpid(child, os.WNOHANG)
        if finished:
            break
        if time.monotonic() >= deadline:
            os.kill(child, 9)
            os.waitpid(child, 0)
            raise AssertionError("owned CPU crash fixture exceeded finite budget")
        time.sleep(0.001)
    assert os.waitstatus_to_exitcode(status) == 73
    stage = spec.out / "games" / "game_00000000.npz.writing"
    assert stage.exists()
    metadata, arrays = fixture.read_game(stage)
    recovered = worker.run_pooled_games(spec, fixture.FakeEvaluator(), fixture.fake_tablebase(), spec.out / "games",
                                       target_rows=2, max_rows=4, batch_wait_ms=1, max_writes=1, resume=True)
    assert len(recovered["receipts"]) == 2
    assert not stage.exists()
    final_metadata, final_arrays = fixture.read_game(stage.with_suffix(""))
    assert metadata == final_metadata
    assert all(np.array_equal(arrays[key], final_arrays[key]) for key in arrays)


def test_foreign_or_incomplete_stage_is_preserved(tmp_path: Path) -> None:
    fixture = fixture_module()
    spec = fixture.spec(tmp_path)
    worker.run_worker(spec, fixture.FakeEvaluator(), fixture.fake_tablebase(), shared_dispatch=(2, 4, 1, 1))
    stage = spec.out / "games" / "game_00000000.npz.writing"
    stage.write_bytes(b"unknown incomplete science")
    (spec.out / "games" / "game_00000000.checkpoint.json").unlink()
    with pytest.raises(ValueError, match="Cannot load file"):
        worker.run_pooled_games(spec, fixture.FakeEvaluator(), fixture.fake_tablebase(), spec.out / "games",
                                target_rows=2, max_rows=4, batch_wait_ms=1, max_writes=1, resume=True)
    assert stage.read_bytes() == b"unknown incomplete science"


def test_invalid_consumer_geometry_closes_created_dispatcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = TeacherDispatcher.close
    closed = []
    def close(self: Any, *, timeout: float) -> None:
        original(self, timeout=timeout)
        assert not self._thread.is_alive()
        closed.append(self)
    monkeypatch.setattr(TeacherDispatcher, "close", close)
    fixture = fixture_module()
    spec = fixture.spec(tmp_path)
    with pytest.raises(ValueError, match="capacities"):
        worker.run_worker(spec, fixture.FakeEvaluator(), fixture.fake_tablebase(), shared_dispatch=(2, 4, 1, 0))
    with pytest.raises(ValueError, match="bounded game label"):
        serve_ceres_stream(io.StringIO(), io.StringIO(), total_games=2, max_games=0,
                           target_rows=2, max_rows=4, batch_wait_ms=1, poll_seconds=0.001, deadline_seconds=5,
                           load_roots=lambda _message: (), infer=lambda roots: roots, publish=lambda _message, _values: {})
    assert len(closed) == 2


def test_broken_rpc_startup_output_closes_both_owners(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = []
    for owner in (TeacherDispatcher, DurableWriter):
        original = owner.close
        def wrap(self: Any, *, timeout: float, close: Any = original) -> None:
            close(self, timeout=timeout)
            assert not self._thread.is_alive()
            closed.append(self)
        monkeypatch.setattr(owner, "close", wrap)
    class BrokenOutput(io.StringIO):
        def write(self, _message: str) -> int:
            raise OSError("closed RPC output")
    with pytest.raises(OSError, match="closed RPC output"):
        serve_ceres_stream(io.StringIO(), BrokenOutput(), total_games=2, max_games=2,
                           target_rows=2, max_rows=4, batch_wait_ms=1, poll_seconds=0.001, deadline_seconds=5,
                           load_roots=lambda _message: (), infer=lambda roots: roots, publish=lambda _message, _values: {})
    assert len(closed) == 2
