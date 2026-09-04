from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest
import zarr

from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.moves.leela_index import compact_index_for_move
from scripts import bt4_raw_corpus_sidecar as tool
from scripts import derive_corpus_targets as derive
from scripts import gen_sf_rooted_corpus as corpus


class FakeSession:
    def run(self, _outputs: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        rows = next(iter(feed.values())).shape[0]
        return [np.zeros((rows, tool.COMPACT_POLICY_SIZE), dtype=np.float32)]


def test_parent_holds_lease_through_child_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    @contextmanager
    def fake_lease(
        _path: Path,
        *,
        poll_seconds: float,
        description: str,
    ):
        assert poll_seconds == 0.25
        assert description == "GPU"
        events.append("lease_acquired")
        try:
            yield
        finally:
            events.append("lease_released")

    class FakeChild:
        exitcode = 0

        def start(self) -> None:
            assert events == ["lease_acquired"]
            events.append("child_started")

        def join(self) -> None:
            events.append("child_exited")

        def is_alive(self) -> bool:
            return False

        def terminate(self) -> None:
            raise AssertionError("successful child must not be terminated")

    class FakeContext:
        def Process(self, *, target: Any, args: tuple[Any, ...]) -> FakeChild:
            assert target is tool.label_child
            assert len(args) == 1
            return FakeChild()

    monkeypatch.setattr(tool, "advisory_lease", fake_lease)
    monkeypatch.setattr(
        tool.multiprocessing,
        "get_context",
        lambda method: FakeContext() if method == "fork" else None,
    )
    args = argparse.Namespace(gpu_lock=Path("lock"), lock_poll_seconds=0.25)
    tool.run_label_child_under_gpu_lease(args)
    assert events == [
        "lease_acquired",
        "child_started",
        "child_exited",
        "lease_released",
    ]


def test_caught_up_preflight_never_acquires_gpu_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    @contextmanager
    def fake_lease(
        _path: Path,
        *,
        poll_seconds: float,
        description: str,
    ):
        assert poll_seconds == 0.25
        events.append(f"{description}_acquired")
        try:
            yield
        finally:
            events.append(f"{description}_released")

    monkeypatch.setattr(tool, "advisory_lease", fake_lease)
    monkeypatch.setattr(tool, "file_sha256", lambda _path: "onnx-sha")
    monkeypatch.setattr(tool, "load_sources", lambda _sources, _out: [object()])
    monkeypatch.setattr(
        tool,
        "pending_shards",
        lambda _sources, *, onnx_sha256: ([], {"source": 7}),
    )
    monkeypatch.setattr(
        tool,
        "run_label_child_under_gpu_lease",
        lambda _args: events.append("gpu_started"),
    )
    args = argparse.Namespace(
        out_root=tmp_path / "out",
        onnx=tmp_path / "teacher.onnx",
        verify_all=False,
        lock_poll_seconds=0.25,
        source=[],
    )
    assert tool.run(args) == 0
    assert events == ["sidecar-writer_acquired", "sidecar-writer_released"]


def test_forked_descriptor_keeps_real_flock_after_parent_context_exits(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "lease.lock"
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    child_pid = -1
    with tool.advisory_lease(
        lock_path,
        poll_seconds=0.01,
        description="test",
    ):
        child_pid = os.fork()
        if child_pid == 0:  # pragma: no cover - assertions run in the parent
            os.close(ready_read)
            os.close(release_write)
            os.write(ready_write, b"1")
            os.read(release_read, 1)
            os._exit(0)
        os.close(ready_write)
        os.close(release_read)
        assert os.read(ready_read, 1) == b"1"

    probe = os.open(lock_path, os.O_RDWR)
    unexpectedly_acquired = False
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            unexpectedly_acquired = True
        except BlockingIOError:
            pass
    finally:
        os.write(release_write, b"1")
        os.close(release_write)
        os.close(ready_read)
        _, status = os.waitpid(child_pid, 0)
        os.close(probe)
    assert not unexpectedly_acquired
    assert os.waitstatus_to_exitcode(status) == 0

    final_probe = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(final_probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(final_probe)


def make_source(tmp_path: Path, *, bad_input_key: bool = False) -> tool.SourceSpec:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    board = chess.Board()
    planes = encode_position(
        board,
        add_features=True,
        input_history_encoding=derive.INPUT_HISTORY_ENCODING,
        input_extra_features=derive.INPUT_EXTRA_FEATURES,
    )
    row = {
        "schema": corpus.ROW_SCHEMA,
        "fen": board.fen(),
        "history_root_fen": board.fen(en_passant="fen"),
        "history_uci": [],
        "game_id": 17,
        "ply": 3,
        "stm": "w",
        "piece_count": 32,
        "input_key": (
            "0" * 32 if bad_input_key else corpus.input_tensor_key(planes)
        ),
        "run": {
            "config_sha256": "config-sha",
            corpus.KEY_HISTORY_REP_FIX: corpus.HISTORY_REP_FIX,
        },
    }
    shard = source_dir / "w00-00000.jsonl.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    inventory = derive.ProgressInventory(
        shards=(shard,),
        shard_rows=(1,),
        rows_claimed=1,
        progress_files=("w00.progress.jsonl",),
        torn_tail_files=(),
        unlisted_on_disk=(),
    )
    out_dir = tmp_path / "sidecars" / "test-source"
    out_dir.mkdir(parents=True)
    return tool.SourceSpec(
        source_id="test-source",
        corpus_dir=source_dir,
        out_dir=out_dir,
        manifest={
            "config_sha256": "config-sha",
            "row_schema": corpus.ROW_SCHEMA,
            corpus.KEY_HISTORY_REP_FIX: corpus.HISTORY_REP_FIX,
        },
        manifest_sha256="manifest-sha",
        inventory=inventory,
    )


def pending_for(source: tool.SourceSpec) -> tool.PendingShard:
    shard = source.inventory.shards[0]
    return tool.PendingShard(
        source=source,
        path=shard,
        claimed_rows=1,
        target=source.out_dir / tool.sidecar_name(shard.name),
    )


def test_source_argument_and_sidecar_names_are_unambiguous() -> None:
    source_id, path = tool.parse_source_arg("run06=/tmp/corpus")
    assert source_id == "run06"
    assert path == Path("/tmp/corpus")
    assert tool.sidecar_name("w03-00014.jsonl.zst") == "w03-00014.bt4.zarr"
    assert tool.sidecar_name("w03-00014.jsonl.gz") == "w03-00014.bt4.zarr"
    with pytest.raises(ValueError, match="ID=PATH"):
        tool.parse_source_arg("../escape=/tmp/corpus")
    with pytest.raises(ValueError, match="unsupported"):
        tool.sidecar_name("w03-00014.jsonl")


def test_receipt_reader_tolerates_only_a_torn_final_line(tmp_path: Path) -> None:
    progress = tmp_path / "progress.jsonl"
    progress.write_bytes(
        b'{"source_shard":"a.jsonl.zst","positions":1}\n'
        b'{"source_shard":"torn.jsonl.zst"'
    )
    assert set(tool.read_receipts(progress)) == {"a.jsonl.zst"}

    progress.write_bytes(
        b'{"source_shard":"a.jsonl.zst","positions":1}\nnot-json\n'
    )
    with pytest.raises(ValueError, match="damaged progress"):
        tool.read_receipts(progress)


def test_receipt_append_repairs_torn_tail_before_adoption(tmp_path: Path) -> None:
    progress = tmp_path / "progress.jsonl"
    progress.write_bytes(
        b'{"source_shard":"a.jsonl.zst","positions":1}\n'
        b'{"source_shard":"lost.jsonl.zst"'
    )
    tool.append_receipt(
        progress,
        {"source_shard": "b.jsonl.zst", "positions": 2},
    )

    assert set(tool.read_receipts(progress)) == {"a.jsonl.zst", "b.jsonl.zst"}
    assert progress.read_bytes().endswith(b"\n")


def test_label_raw_shard_verifies_history_and_publishes_joinable_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_source(tmp_path)
    pending = pending_for(source)
    monkeypatch.setattr(
        tool,
        "remap_provenance",
        lambda: {"commit": "c", "dirty": False, "blobs": {"x": "y"}},
    )

    attrs = tool.label_shard(
        pending,
        sess=FakeSession(),
        input_name="input",
        input_dtype=np.dtype(np.float32),
        providers=["fake"],
        policy_name="policy",
        onnx_path=tmp_path / "fake.onnx",
        onnx_sha256="onnx-sha",
        remap_stamp={"commit": "c", "dirty": False, "blobs": {"x": "y"}},
        batch_size=1,
    )

    assert pending.target.is_dir()
    assert not pending.target.with_name(pending.target.name + ".writing").exists()
    group: Any = zarr.open_group(str(pending.target), mode="r")
    policy = np.asarray(group[tool.POLICY_FIELD][:])
    legal = {
        compact_index_for_move(chess.Board(), move)
        for move in chess.Board().legal_moves
    }
    assert policy.shape == (1, tool.COMPACT_POLICY_SIZE)
    assert float(policy.sum()) == pytest.approx(1.0)
    assert set(np.flatnonzero(policy[0]).tolist()) == legal
    assert np.asarray(group[tool.GAME_ID_FIELD][:]).tolist() == [17]
    assert np.asarray(group[tool.PLY_FIELD][:]).tolist() == [3]
    assert attrs["teacher_evaluations_per_position"] == 1
    assert attrs["search_nodes"] == 0
    assert attrs["source_sha256"] == tool.file_sha256(source.inventory.shards[0])
    tool.validate_existing(
        pending,
        onnx_sha256="onnx-sha",
        policy_output="policy",
        providers=["fake"],
    )
    tool.verify_shard(
        pending,
        onnx_sha256="onnx-sha",
        expected_policy_output="policy",
        expected_providers=["fake"],
        expected_remap={"commit": "c", "dirty": False, "blobs": {"x": "y"}},
        batch_size=1,
    )


def test_label_raw_shard_refuses_wrong_banked_input_key(tmp_path: Path) -> None:
    source = make_source(tmp_path, bad_input_key=True)
    pending = pending_for(source)
    with pytest.raises(ValueError, match="encoded input key"):
        tool.label_shard(
            pending,
            sess=FakeSession(),
            input_name="input",
            input_dtype=np.dtype(np.float32),
            providers=["fake"],
            policy_name="policy",
            onnx_path=tmp_path / "fake.onnx",
            onnx_sha256="onnx-sha",
            remap_stamp={"commit": "c", "dirty": False, "blobs": {"x": "y"}},
            batch_size=1,
        )
    assert not pending.target.exists()


def test_pending_shards_never_adopts_unlisted_in_flight_file(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    in_flight = source.corpus_dir / "w00-00001.jsonl.gz"
    in_flight.write_bytes(b"still being written")

    pending, complete = tool.pending_shards([source], onnx_sha256="onnx-sha")

    assert [item.path.name for item in pending] == ["w00-00000.jsonl.gz"]
    assert complete == {"test-source": 0}
    assert tool.sidecar_name(in_flight.name) not in {
        item.target.name for item in pending
    }


def test_verify_marks_live_snapshot_and_overwrites_stale_pass_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_source(tmp_path)
    pending = pending_for(source)
    remap = {"commit": "c", "dirty": False, "blobs": {"x": "y"}}
    attrs = tool.label_shard(
        pending,
        sess=FakeSession(),
        input_name="input",
        input_dtype=np.dtype(np.float32),
        providers=["fake"],
        policy_name="policy",
        onnx_path=tmp_path / "fake.onnx",
        onnx_sha256="onnx-sha",
        remap_stamp=remap,
        batch_size=1,
    )
    tool.append_receipt(
        source.out_dir / tool.PROGRESS_NAME,
        tool.receipt_from_attrs(attrs, pending.target),
    )
    args = argparse.Namespace(source=[], batch_size=1, gpu_mem_gb=0.0)
    out_root = source.out_dir.parent
    monkeypatch.setattr(tool, "remap_provenance", lambda: remap)
    monkeypatch.setattr(tool, "load_sources", lambda *_args: [source])

    assert tool.verify_all(args, out_root=out_root, onnx_sha="onnx-sha") == 0
    receipt = json.loads((out_root / tool.VERIFY_NAME).read_text(encoding="utf-8"))
    assert receipt["verdict"] == "SNAPSHOT_PASS"
    assert receipt["snapshot_only"] is True

    final_source = replace(source, corpus_complete=True)
    monkeypatch.setattr(tool, "load_sources", lambda *_args: [final_source])
    assert tool.verify_all(args, out_root=out_root, onnx_sha="onnx-sha") == 0
    receipt = json.loads((out_root / tool.VERIFY_NAME).read_text(encoding="utf-8"))
    assert receipt["verdict"] == "PASS"
    assert receipt["snapshot_only"] is False

    group: Any = zarr.open_group(str(pending.target), mode="a")
    group[tool.POLICY_FIELD][:] = np.zeros_like(group[tool.POLICY_FIELD][:])
    with pytest.raises(ValueError, match="unnormalized stored policy"):
        tool.verify_all(args, out_root=out_root, onnx_sha="onnx-sha")
    receipt = json.loads((out_root / tool.VERIFY_NAME).read_text(encoding="utf-8"))
    assert receipt["verdict"] == "FAIL"
    assert receipt["error_type"] == "ValueError"
