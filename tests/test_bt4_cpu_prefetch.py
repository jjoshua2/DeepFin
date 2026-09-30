"""CPU-only scheduling and failure contracts for opt-in raw BT4 prefetch."""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
import os
import signal
import threading
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import bt4_raw_corpus_sidecar as tool
from tests.test_bt4_raw_wdl import REMAP, Session, source3


def label(pending, session, *, prefetch: bool, dtype="float32", batch_size=2):
    return tool.label_shard(
        pending, sess=session, input_name="input", input_dtype=np.dtype(dtype),
        providers=["fixture"], policy_name="policy",
        onnx_path=pending.path.parent / "fake.onnx", onnx_sha256="teacher",
        remap_stamp=REMAP, batch_size=batch_size, cpu_prefetch=prefetch,
        wdl_output={"output": "value_head", "kind": "logits", "dtype": "float16"},
    )


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_prefetch_exact_feeds_outputs_identity_and_partial_batch(
    tmp_path: Path, dtype: str,
) -> None:
    _, pending = source3(tmp_path)
    caller = threading.get_ident()

    class RecordingSession(Session):
        def __init__(self):
            super().__init__()
            self.feeds = []

        def run(self, names, feed):
            assert threading.get_ident() == caller
            self.feeds.append(feed["input"].copy())
            return super().run(names, feed)

    sessions = [RecordingSession(), RecordingSession()]
    targets = [replace(pending, target=pending.target.with_name(name))
               for name in ("serial.zarr", "prefetch.zarr")]
    attrs = [label(target, session, prefetch=enabled, dtype=dtype)
             for target, session, enabled in zip(targets, sessions, (False, True), strict=True)]
    assert [a["cpu_prefetch"] for a in attrs] == [False, True]
    for a in attrs:
        a.pop("cpu_prefetch")
        a.pop("published_unix")
    assert attrs[0] == attrs[1]
    assert [len(x) for x in sessions[0].feeds] == [2, 1]
    for a, b in zip(sessions[0].feeds, sessions[1].feeds, strict=True):
        assert a.dtype == b.dtype == np.dtype(dtype)
        assert a.tobytes() == b.tobytes()
    groups: list[Any] = [zarr.open_group(str(t.target), mode="r") for t in targets]
    assert set(groups[0].array_keys()) == set(groups[1].array_keys())
    for key in groups[0].array_keys():
        assert np.asarray(groups[0][key][:]).tobytes() == np.asarray(groups[1][key][:]).tobytes()
    for target in targets:
        tool.verify_shard(
            target, onnx_sha256="teacher", expected_policy_output="policy",
            expected_providers=["fixture"], expected_remap=REMAP, batch_size=2,
            expected_wdl={"output": "value_head", "kind": "logits", "dtype": "float16"},
        )


def test_cpu_prepares_exactly_one_ahead_during_main_thread_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, pending = source3(tmp_path)
    caller = threading.get_ident()
    entered_inference = threading.Event()
    second_prepared = threading.Event()
    prepared: list[int] = []
    workers: set[threading.Thread] = set()
    original = tool._prepare_input_batch

    def prepare(rows, **kwargs):
        workers.add(threading.current_thread())
        assert threading.get_ident() != caller
        key = int(rows[0]["game_id"])
        prepared.append(key)
        if key == 18:
            assert entered_inference.wait(5)
        value = original(rows, **kwargs)
        if key == 18:
            second_prepared.set()
        return value

    class OverlapSession(Session):
        def run(self, names, feed):
            assert threading.get_ident() == caller
            if self.cursor == 0:
                entered_inference.set()
                assert second_prepared.wait(5)
                # No third batch is scheduled until the consumer asks again.
                assert prepared == [17, 18]
            return super().run(names, feed)

    monkeypatch.setattr(tool, "_prepare_input_batch", prepare)
    label(pending, OverlapSession(), prefetch=True, batch_size=1)
    assert prepared == [17, 18, 19]
    assert len(workers) == 1
    assert all(not worker.is_alive() for worker in workers)


@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt])
def test_consumer_failure_closes_reader_and_joins_owned_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException],
) -> None:
    _, pending = source3(tmp_path)
    original = tool.iter_bt4_input_rows
    closed = threading.Event()
    workers: set[threading.Thread] = set()
    read_count = 0

    def tracked(path):
        nonlocal read_count
        workers.add(threading.current_thread())
        try:
            for row in original(path):
                read_count += 1
                yield row
        finally:
            closed.set()

    class FailingSession(Session):
        def run(self, names, feed):
            del names, feed
            raise failure("consumer failed")

    monkeypatch.setattr(tool, "iter_bt4_input_rows", tracked)
    with pytest.raises(failure, match="consumer failed"):
        label(pending, FailingSession(), prefetch=True, batch_size=1)
    assert closed.is_set()
    assert read_count <= 2
    assert all(not worker.is_alive() for worker in workers)
    assert not pending.target.exists()
    assert not pending.target.with_name(pending.target.name + ".writing").exists()


def test_producer_error_is_propagated_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, pending = source3(tmp_path)
    original = tool.iter_bt4_input_rows
    closed = threading.Event()
    workers: set[threading.Thread] = set()

    def broken(path):
        workers.add(threading.current_thread())
        try:
            with tool.closing(original(path)) as reader:
                yield next(reader)
                raise ValueError("bad next row")
        finally:
            closed.set()

    monkeypatch.setattr(tool, "iter_bt4_input_rows", broken)
    session = Session()
    with pytest.raises(ValueError, match="bad next row"):
        label(pending, session, prefetch=True, batch_size=1)
    assert session.cursor == 1
    assert closed.is_set()
    assert all(not worker.is_alive() for worker in workers)
    assert not pending.target.exists()


def test_interrupt_waiting_for_first_batch_stops_reader_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, pending = source3(tmp_path)
    original = tool.iter_bt4_input_rows
    started, release, closed = (threading.Event() for _ in range(3))
    workers: set[threading.Thread] = set()

    def reader(path):
        workers.add(threading.current_thread())
        try:
            started.set()
            assert release.wait(5)
            yield from original(path)
        finally:
            closed.set()

    def interrupted_result(_future, timeout=None):
        del timeout
        assert started.wait(5)
        release.set()
        raise KeyboardInterrupt("waiting for input")

    monkeypatch.setattr(tool, "iter_bt4_input_rows", reader)
    monkeypatch.setattr(Future, "result", interrupted_result)
    with pytest.raises(KeyboardInterrupt, match="waiting for input"):
        label(pending, Session(), prefetch=True, batch_size=1)
    assert closed.is_set()
    assert all(not worker.is_alive() for worker in workers)
    assert not pending.target.exists()


@pytest.mark.parametrize("prefetch", [False, True])
def test_json_null_row_is_not_mistaken_for_end_of_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefetch: bool,
) -> None:
    _, pending = source3(tmp_path)
    original = tool.iter_bt4_input_rows

    def invalid(path):
        with tool.closing(original(path)) as reader:
            yield next(reader)
            yield None

    monkeypatch.setattr(tool, "iter_bt4_input_rows", invalid)
    with pytest.raises(AttributeError):
        label(replace(pending, claimed_rows=2), Session(), prefetch=prefetch)
    assert not pending.target.exists()


def test_cli_default_off_and_run_group_passes_enabled_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, pending = source3(tmp_path)
    argv = ["--source", "x=" + str(source.corpus_dir), "--out-root",
            str(source.out_dir.parent), "--gpu-lock", str(tmp_path / "lock"),
            "--gpu-mem-gb", "0", "--min-free-gib", "0", "--batch-size", "128"]
    assert tool.build_parser().parse_args(argv).cpu_prefetch is False
    args = tool.build_parser().parse_args([*argv, "--cpu-prefetch"])
    monkeypatch.setattr(tool, "load_sources", lambda *_: [source])
    monkeypatch.setattr(tool, "open_session",
                        lambda *a, **k: (Session(), "input", np.dtype("float32"), ["fixture"]))
    monkeypatch.setattr(tool, "file_sha256", lambda _: "teacher")
    monkeypatch.setattr(tool, "remap_provenance", lambda: REMAP)
    monkeypatch.setattr(tool, "pending_shards", lambda *a, **k: ([pending], {}))

    def selected(target, **kwargs):
        assert target is pending
        assert kwargs["cpu_prefetch"] is True
        assert kwargs["batch_size"] == 128
        raise RuntimeError("configuration reached label_shard")

    monkeypatch.setattr(tool, "label_shard", selected)
    with pytest.raises(RuntimeError, match="configuration reached label_shard"):
        tool.run_label_group(args)


def test_sigint_during_shutdown_waits_for_worker_before_reader_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, pending = source3(tmp_path)
    started, release, finished = (threading.Event() for _ in range(3))
    workers: set[threading.Thread] = set()
    old_handler = signal.getsignal(signal.SIGINT)

    def preparing(*args, **kwargs):
        del args, kwargs
        workers.add(threading.current_thread())
        try:
            yield object()
            started.set()
            assert release.wait(5)
            yield object()
        finally:
            finished.set()

    original_shutdown = tool.ThreadPoolExecutor.shutdown

    def shutdown(executor, **kwargs):
        assert started.wait(5)
        # Real process SIGINT at the formerly vulnerable shutdown boundary.
        os.kill(os.getpid(), signal.SIGINT)
        release.set()
        original_shutdown(executor, **kwargs)

    monkeypatch.setattr(tool, "_iter_prepared_inputs", preparing)
    monkeypatch.setattr(tool.ThreadPoolExecutor, "shutdown", shutdown)
    def early_exit():
        with tool.prepared_input_batches(
            pending, input_dtype=np.dtype("float32"), batch_size=1, cpu_prefetch=True,
        ) as batches:
            next(batches)
            assert started.wait(5)

    with pytest.raises(KeyboardInterrupt):
        early_exit()
    assert finished.is_set()
    assert all(not worker.is_alive() for worker in workers)
    assert signal.getsignal(signal.SIGINT) is old_handler
