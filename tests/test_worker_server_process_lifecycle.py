"""Ephemeral process-level qualification for the DeepFin worker/server path."""
from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import socket
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import requests

from chess_anti_engine.replay.shard import ShardMeta, load_shard_arrays, samples_to_arrays, save_local_shard_arrays
from chess_anti_engine.version import PROTOCOL_VERSION
from chess_anti_engine.worker import WorkerSession

TRIAL = "trial_00000"


def _serve(sock: socket.socket, root: str, book: str, compact_size: int) -> None:
    import uvicorn
    from chess_anti_engine.server.app import create_app
    app = create_app(server_root=root, users_db="users.json", opening_book_path=book,
                     upload_compact_shard_size=compact_size, upload_compact_max_age_seconds=3600)
    uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on")).run(sockets=[sock])


class _Server:
    def __init__(self, root: Path, book: Path, compact_size: int = 2) -> None:
        self.root, self.book = root, book
        self.compact_size = compact_size
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(128)
        self.port = self.sock.getsockname()[1]
        self.proc = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.proc = multiprocessing.get_context("fork").Process(
            target=_serve, args=(self.sock, str(self.root), str(self.book), self.compact_size), daemon=True)
        self.proc.start()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.proc.exitcode is not None:
                raise RuntimeError(f"fixture server exited with {self.proc.exitcode}")
            try:
                r = requests.get(self.url + f"/v1/trials/{TRIAL}/manifest", timeout=.25,
                                 headers={"X-CAE-Worker-Version": "0.0.0",
                                          "X-CAE-Protocol-Version": str(PROTOCOL_VERSION)})
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(.05)
        raise TimeoutError("fixture server startup timed out")

    def stop(self, *, crash: bool = False) -> None:
        if self.proc is not None:
            self.proc.kill() if crash else self.proc.terminate()
            self.proc.join(10)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join(5)
            self.proc = None

    def close(self) -> None:
        self.stop()
        self.sock.close()


class _DropFirstAck:
    def __init__(self) -> None:
        self.http = requests.Session()
        self.drop = True

    def get(self, *args, **kwargs):
        return self.http.get(*args, **kwargs)

    def post(self, *args, **kwargs):
        response = self.http.post(*args, **kwargs)
        if self.drop:
            self.drop = False
            response.close()
            raise requests.ConnectionError("injected lost ACK after server commit")
        return response

    def close(self) -> None:
        self.http.close()


def _worker(url: str, cache: Path, pending: Path, transport) -> WorkerSession:
    w = object.__new__(WorkerSession)
    w.server, w.log = url, logging.getLogger("test.process_lifecycle")
    w.args = SimpleNamespace(poll_seconds=0, self_update=False, inference_slot_name="",
                             inference_slot_input_planes=146, username="u", password="fixture-only")
    w.cfg = {}
    w.fixed_trial_id = w.leased_trial_id = TRIAL
    w.trial_api_prefix = f"/v1/trials/{TRIAL}"
    w.lease_id, w.machine_id, w.worker_id = "", "fixture", "fixture"
    w._requests = transport
    w._manifest_poll_failures = 0
    w.manifest_state, w.manifest_state_elapsed_s = "active", None
    w.pause_selfplay_active = w._hold_on_pause = False
    cast(Any, w).inference_client = SimpleNamespace(input_planes=146)
    w.pending_dir = pending
    pending.mkdir(parents=True, exist_ok=True)
    w._pending_upload_lock = threading.Lock()
    w._pending_buffer_flushes, w._pending_buffer_positions = deque(), 0
    w._upload_buf_lock = None
    cast(Any, w).upload_buf = SimpleNamespace(positions=0, model_sha=None, model_step=0)
    w.last_successful_send_s = 0
    w._upload_pending_arena_results = lambda: None
    w.cache_dir = cache
    cache.mkdir(parents=True, exist_ok=True)
    w.model_sha, w.model_step, w.last_model_sha = "", 0, None
    w.model = w.model_cfg_active = None
    w._direct_evaluator, w._evaluator_model_id = None, None
    w._slot_planes_unknown_warned = False
    cast(Any, w)._load_and_compile_model = lambda path, cfg, **_kw: (path, cfg)
    w._resync_evaluator_to_model = lambda: None
    w.opening_book_path = w.opening_book_path_2 = w.opening_fen_list_path = None
    w.last_ob_sha = w.last_ob2_sha = w.last_fenlist_sha = None
    w._auth = ("u", "fixture-only")
    return w


def _write_shard(path: Path, rows: int = 2) -> None:
    from tests.test_server_upload_security import _sample
    save_local_shard_arrays(
        path, arrs=samples_to_arrays([_sample(i + 1) for i in range(rows)]),
        meta=ShardMeta(username="u", run_id=TRIAL, games=1, positions=rows,
                       model_sha256="f" * 64, model_step=7))


@pytest.mark.skipif(os.name != "posix", reason="this qualification starts a forked loopback server")
def test_process_worker_cold_assets_lost_ack_restart_retry(tmp_path: Path) -> None:
    from chess_anti_engine.server.auth import UserRecord, hash_password, save_users
    root = tmp_path / "server"
    pub = root / "trials" / TRIAL / "publish"
    pub.mkdir(parents=True)
    salt, pw_hash, iterations = hash_password("fixture-only")
    save_users(root / "users.json", {"u": UserRecord(username="u", salt_b64=salt,
                                                       hash_b64=pw_hash, iterations=iterations)})
    model_bytes = b"fixture model payload; evaluator load is stubbed"
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    (pub / "latest_model.pt").write_bytes(model_bytes)
    fen_bytes = b"8/8/8/8/8/8/8/K6k w - - 0 1\n"
    (pub / "opening_fen_list_live.txt").write_bytes(fen_bytes)
    book = tmp_path / "book.bin"
    book.write_bytes(b"fixture opening book")
    book_sha = hashlib.sha256(book.read_bytes()).hexdigest()
    manifest = {
        "protocol_version": PROTOCOL_VERSION, "task": {"type": "arena"}, "trainer_step": 7,
        "model_config": {"input_extra_features": "v1"},
        "model": {"sha256": model_sha, "endpoint": f"/v1/trials/{TRIAL}/model"},
        "opening_book": {"filename": book.name, "sha256": book_sha,
                         "endpoint": "/v1/opening_book"},
        "opening_fen_list": {"filename": "opening_fen_list_live.txt",
                              "sha256": hashlib.sha256(fen_bytes).hexdigest(),
                              "endpoint": f"/v1/trials/{TRIAL}/opening_fen_list"},
    }
    (pub / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    server = _Server(root, book, compact_size=100)
    worker = None
    try:
        server.start()
        cache, pending = tmp_path / "cache", tmp_path / "worker" / "pending"
        worker = _worker(server.url, cache, pending, requests.Session())

        # Actual production poll -> asset sync against a cold cache.
        polled = WorkerSession._poll_manifest(worker)
        assert polled is not None
        assert polled["model"]["sha256"] == model_sha
        WorkerSession._sync_assets(worker, polled)
        assert (cache / f"model_{model_sha}.pt").read_bytes() == model_bytes
        assert worker.opening_book_path is not None
        assert worker.opening_fen_list_path is not None
        assert Path(worker.opening_book_path).read_bytes() == b"fixture opening book"
        assert Path(worker.opening_fen_list_path).read_bytes() == fen_bytes

        # Interrupt after durable pending promotion but before compaction.
        # Restart recovery should recognize the digest and avoid duplicate rows.
        shard = pending / "retry.zarr"
        _write_shard(shard)
        lost_ack = _DropFirstAck()
        worker._requests.close()
        worker._requests = lost_ack
        with pytest.raises(requests.ConnectionError, match="lost ACK"):
            WorkerSession._upload_pending_shards_locked(worker, default_elapsed_s=1)
        assert shard.is_dir(), "the worker retains a shard when no ACK arrives"
        compacted = root / "trials" / TRIAL / "inbox" / "_compacted"
        pending_server = root / "trials" / TRIAL / "inbox" / "_pending"
        assert len(list(pending_server.glob("*.zarr"))) == 1
        assert not list(compacted.glob("*.zarr"))

        # Restart the server and retry. The digest recovered from the pending
        # filename makes this pre-compaction retry idempotent.
        server.stop(crash=True)
        server.start()
        worker._requests.close()
        worker._requests = requests.Session()
        WorkerSession._upload_pending_shards_locked(worker, default_elapsed_s=1)
        assert not shard.exists()
        assert len(list(pending_server.glob("*.zarr"))) == 1
        assert not list(compacted.glob("*.zarr"))

        # Add one row to compact the recovered two. Drop its ACK after the
        # compaction commit, restart, and retry. A completed compaction no longer
        # persists the source digest, so this narrow ambiguous-commit case is
        # at-least-once and can duplicate rows.
        server.stop(crash=True)
        server.compact_size = 3
        server.start()
        second = pending / "retry_after_compact.zarr"
        _write_shard(second, rows=1)
        lose_compacted_ack = _DropFirstAck()
        worker._requests.close()
        worker._requests = lose_compacted_ack
        with pytest.raises(requests.ConnectionError, match="lost ACK"):
            WorkerSession._upload_pending_shards_locked(worker, default_elapsed_s=1)
        assert second.is_dir()
        committed = sorted(compacted.glob("*.zarr"))
        assert len(committed) == 1
        assert load_shard_arrays(committed[0])[0]["x"].shape[0] == 3

        server.stop(crash=True)
        server.start()
        worker._requests.close()
        worker._requests = requests.Session()
        WorkerSession._upload_pending_shards_locked(worker, default_elapsed_s=1)
        assert not second.exists()
        third = pending / "new_after_restart.zarr"
        _write_shard(third, rows=1)
        WorkerSession._upload_pending_shards_locked(worker, default_elapsed_s=1)
        assert not third.exists()
        fourth = pending / "new_after_restart_2.zarr"
        _write_shard(fourth, rows=1)
        WorkerSession._upload_pending_shards_locked(worker, default_elapsed_s=1)
        assert not fourth.exists()
        committed = sorted(compacted.glob("*.zarr"))
        assert sum(load_shard_arrays(p)[0]["x"].shape[0] for p in committed) == 6
    finally:
        if worker is not None:
            worker._requests.close()
        server.close()
