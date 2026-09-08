"""Tiny real derived/shuffled storage, fake named-output teacher, no inference."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import zarr

from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.encoding.lc0 import x_to_lc0_planes
from scripts import bt4_derived_wdl_sidecar as tool
from scripts import derive_corpus_targets as derive
from tests.test_sf_policy_rewrite import fixture


class Session:
    def __init__(self, bad: str | None = None):
        self.bad = bad
        self.inputs: list[np.ndarray] = []
        self.values: list[np.ndarray] = []
        self.requests: list[list[str]] = []

    def get_inputs(self):
        return [
            SimpleNamespace(name="input", type="tensor(float)", shape=[None, 112, 8, 8])
        ]

    def get_outputs(self):
        return [SimpleNamespace(name="value", type="tensor(float16)", shape=[None, 3])]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, names, feed):
        self.requests.append(names)
        self.inputs.append(feed["input"].copy())
        n = len(feed["input"])
        result = np.tile([0.125, 0.25, 0.625], (n, 1)).astype("float16")
        if self.bad == "nan":
            result[0, 0] = np.nan
        elif self.bad == "mass":
            result[:] = 0.5
        elif self.bad == "shape":
            result = result[:, :2]
        elif self.bad == "dtype":
            result = result.astype("float32")
        self.values.append(result.copy())
        return [] if self.bad == "missing" else [result]


def setup(tmp_path: Path):
    _raw, source, rows = fixture(tmp_path)
    model = tmp_path / "teacher.onnx"
    model.write_bytes(b"fake explicit output graph")
    args = tool.build_parser().parse_args(
        [
            "--source",
            str(source),
            "--out",
            str(tmp_path / "labels"),
            "--onnx",
            str(model),
            "--expected-source-summary-sha256",
            tool.file_sha256(source / tool.SUMMARY),
            "--expected-onnx-sha256",
            tool.file_sha256(model),
            "--wdl-output",
            "value",
            "--wdl-output-kind",
            "probabilities",
            "--max-shards",
            "1",
            "--batch-size",
            "2",
            "--minimum-free-gib",
            "0",
        ]
    )
    Path(args.out).mkdir()
    args.invocation = str(tmp_path / "invocation")
    Path(args.invocation).mkdir()
    return args, rows


def install(monkeypatch: pytest.MonkeyPatch, session: Session):
    monkeypatch.setattr(
        tool,
        "open_session",
        lambda *_a, **_k: (
            session,
            "input",
            np.dtype("float32"),
            session.get_providers(),
        ),
    )


def test_actual_selected_shuffled_rows_and_exact_original_feed(tmp_path, monkeypatch):
    args, rows = setup(tmp_path)
    session = Session()
    install(monkeypatch, session)
    tool.produce(args)
    assert session.requests == [["value"], ["value"]]
    source: Any = zarr.open_group(
        str(Path(args.source) / "shard_000000.zarr"), mode="r"
    )
    output: Any = zarr.open_group(str(Path(args.out) / "shard_000000.zarr"), mode="r")
    assert not (Path(args.out) / "shard_000001.zarr").exists()
    assert set(output.array_keys()) == {
        "bt4_wdl_raw",
        "row_index",
        "game_id",
        "ply_index",
        "lc0_feed_sha256",
    }
    np.testing.assert_array_equal(
        output["bt4_wdl_raw"][:], np.concatenate(session.values)
    )
    np.testing.assert_array_equal(output["game_id"][:], source["game_id"][:])
    np.testing.assert_array_equal(output["ply_index"][:], source["ply_index"][:])
    np.testing.assert_array_equal(output["row_index"][:], np.arange(4))
    actual = np.concatenate(session.inputs)
    for i, game in enumerate(source["game_id"][:]):
        row = next(r for r in rows if r["game_id"] == game)
        board = derive.board_from_row(row)
        x = encode_position(
            board,
            input_history_encoding=tool.HISTORY,
            input_extra_features="v2_threats",
        )
        np.testing.assert_array_equal(
            actual[i], x_to_lc0_planes(x, input_history_encoding=tool.HISTORY)
        )
    np.testing.assert_array_equal(
        output["lc0_feed_sha256"][:], tool.row_digests(actual)
    )
    assert "No raw-history replay" in output.attrs["binding"]["history_lineage"]
    before = tool.storage_identity(Path(args.out) / "shard_000000.zarr")

    def forbidden(*_a, **_k):
        pytest.fail("matching complete output must not create a session")

    monkeypatch.setattr(tool, "open_session", forbidden)
    tool.produce(args)
    assert tool.storage_identity(Path(args.out) / "shard_000000.zarr") == before
    args.wdl_output_kind = "logits"
    with pytest.raises(ValueError, match="binding differs"):
        tool.produce(args)


def test_all_101_counter_values_survive_stored_roundtrip():
    original = np.zeros((101, 175, 8, 8), dtype=np.float32)
    original[:, 109] = np.arange(101, dtype=np.float32)[:, None, None] / 100
    original[:, 110] = 0.37
    original[:, 111] = 1
    expected = x_to_lc0_planes(original, input_history_encoding=tool.HISTORY)
    np.testing.assert_array_equal(
        tool.stored_feed(original.astype("float16")), expected
    )


@pytest.mark.parametrize("bad", ["nan", "mass", "shape", "dtype", "missing"])
def test_malformed_output_never_publishes(tmp_path, monkeypatch, bad):
    args, _ = setup(tmp_path)
    install(monkeypatch, Session(bad))
    with pytest.raises(ValueError, match=r"WDL|missing requested"):
        tool.produce(args)
    assert not (Path(args.out) / "shard_000000.zarr").exists()
    assert (Path(args.out) / "shard_000000.zarr.writing").exists()
    with pytest.raises(ValueError, match="partial sidecar"):
        tool.produce(args)


@pytest.mark.parametrize("bad", ["binary", "counter", "presence", "history", "chunk"])
def test_malformed_source_refuses_before_inference(tmp_path, monkeypatch, bad):
    args, _ = setup(tmp_path)
    source: Any = zarr.open_group(
        str(Path(args.source) / "shard_000000.zarr"), mode="a"
    )
    if bad == "binary":
        source["x"][0, 0, 0, 0] = 0.5
    elif bad == "counter":
        source["x"][0, 109] = 0.123
    elif bad == "presence":
        source["has_game_id"][0] = 0
    elif bad == "history":
        source.attrs["history_rep_fix"] = False
    else:
        del source["x"].chunk_store[source["x"]._chunk_key((0, 0, 0, 0))]
    session = Session()
    install(monkeypatch, session)
    with pytest.raises(
        ValueError, match=r"nonbinary|rule50|missing row|source shard|missing stored"
    ):
        tool.produce(args)
    assert not session.requests
    assert not (Path(args.out) / "shard_000000.zarr").exists()


@pytest.mark.parametrize("bad", ["source", "payload", "model", "summary"])
def test_resume_refuses_changed_inputs_or_sidecar(tmp_path, monkeypatch, bad):
    args, _ = setup(tmp_path)
    install(monkeypatch, Session())
    tool.produce(args)
    group: Any
    if bad == "source":
        group = zarr.open_group(str(Path(args.source) / "shard_000000.zarr"), mode="a")
        group["game_id"][0] += 1
    elif bad == "payload":
        group = zarr.open_group(str(Path(args.out) / "shard_000000.zarr"), mode="a")
        group["bt4_wdl_raw"][0] = [0.25, 0.25, 0.5]
    elif bad == "model":
        Path(args.onnx).write_bytes(b"different")
    else:
        path = Path(args.source) / tool.SUMMARY
        path.write_text(path.read_text() + "\n")
    with pytest.raises(
        ValueError, match=r"binding differs|content differs|teacher SHA|summary pin"
    ):
        tool.produce(args)


def test_source_mutation_during_teacher_call_retains_partial(tmp_path, monkeypatch):
    args, _ = setup(tmp_path)
    session = Session()
    original = session.run

    def changed(names, feed):
        result = original(names, feed)
        group: Any = zarr.open_group(
            str(Path(args.source) / "shard_000000.zarr"), mode="a"
        )
        group["ply_index"][0] += 1
        return result

    monkeypatch.setattr(session, "run", changed)
    install(monkeypatch, session)
    with pytest.raises(ValueError, match="source changed"):
        tool.produce(args)
    assert not (Path(args.out) / "shard_000000.zarr").exists()


def test_owned_child_stop_and_completed_receipt(tmp_path, monkeypatch):
    args, _ = setup(tmp_path)
    install(monkeypatch, Session())
    assert tool.run(args) == 0
    completed = list((Path(args.out) / "invocations").glob("*/completed.json"))
    assert len(completed) == 1
    assert json.loads(completed[0].read_text())["rows"] == 4
    (Path(args.out) / "STOP").touch()
    with pytest.raises(ValueError, match="STOP"):
        tool.run(args)
    assert len(list((Path(args.out) / "invocations").glob("*/failed.json"))) == 1


def test_stop_during_owned_teacher_call_reaps_child(tmp_path, monkeypatch):
    import os
    import threading
    import time

    args, _ = setup(tmp_path)
    entered = tmp_path / "entered"
    session = Session()

    def blocked(_names, _feed):
        entered.write_text(str(os.getpid()))
        time.sleep(30)
        pytest.fail("STOP should terminate the owned session process")

    monkeypatch.setattr(session, "run", blocked)
    install(monkeypatch, session)

    def stop_after_entry():
        until = time.monotonic() + 10
        while not entered.exists() and time.monotonic() < until:
            time.sleep(0.01)
        (Path(args.out) / "STOP").touch()

    thread = threading.Thread(target=stop_after_entry)
    thread.start()
    try:
        with pytest.raises(ValueError, match="STOP"):
            tool.run(args)
    finally:
        thread.join(timeout=12)
    assert entered.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(entered.read_text()), 0)
    assert not list((Path(args.out) / "invocations").glob("*/completed.json"))
    assert not (Path(args.out) / "shard_000000.zarr").exists()


def test_gpu_requires_explicit_shared_lease_path(tmp_path):
    args, _ = setup(tmp_path)
    args.gpu_mem_gb = 4
    with pytest.raises(ValueError, match="absolute shared"):
        tool.run(args)


def test_gpu_lease_retained_until_disposable_child_exits(tmp_path, monkeypatch):
    import fcntl
    import multiprocessing
    import time

    args, _ = setup(tmp_path)
    args.gpu_mem_gb = 1
    args.gpu_lock = str(tmp_path / "shared-gpu.lock")
    session = Session()
    monkeypatch.setattr(
        session,
        "get_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    install(monkeypatch, session)
    ready = tmp_path / "producer-returned"
    finish = tmp_path / "allow-exit"
    original = tool.produce

    def held_after_produce(options):
        original(options)
        ready.touch()
        until = time.monotonic() + 10
        while not finish.exists() and time.monotonic() < until:
            time.sleep(0.01)

    monkeypatch.setattr(tool, "produce", held_after_produce)
    process = multiprocessing.get_context("fork").Process(
        target=tool.child, args=(args,)
    )
    process.start()
    try:
        until = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < until:
            time.sleep(0.01)
        assert ready.exists()
        with open(args.gpu_lock, "a") as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finish.touch()
            process.join(timeout=5)
            assert process.exitcode == 0
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        if process.is_alive():
            process.kill()
        process.join()


@pytest.mark.parametrize("column", ["game_id", "ply_index"])
def test_boolean_identity_refused_before_inference(tmp_path, monkeypatch, column):
    args, _ = setup(tmp_path)
    group: Any = zarr.open_group(str(Path(args.source) / "shard_000000.zarr"), mode="a")
    del group[column]
    group.create_dataset(column, data=np.ones(4, dtype=np.bool_), chunks=(4,))
    session = Session()
    install(monkeypatch, session)
    with pytest.raises(ValueError, match="nonintegral source identity"):
        tool.produce(args)
    assert session.requests == []
    assert not (Path(args.out) / "shard_000000.zarr").exists()
