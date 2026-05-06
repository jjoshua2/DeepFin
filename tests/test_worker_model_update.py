from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from chess_anti_engine.model import ModelConfig
import chess_anti_engine.worker as worker_mod
from chess_anti_engine.worker import WorkerSession
from chess_anti_engine.worker_buffer import _BufferedUpload


def _bare_worker_session() -> WorkerSession:
    session = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.worker_model_update")
    session.leased_trial_id = "trial_00000"
    session.pause_selfplay_active = False
    session._stop_selfplay = False
    session._active_reco = {k: None for k in WorkerSession._RECO_RESTART_KEYS}
    session._active_reco["sf_nodes"] = 100
    session._last_manifest_poll_s = time.time()
    session._manifest_mtime = None
    session.model_sha = "old-sha"
    return session


def test_mtime_model_update_swaps_before_reco_restart(tmp_path: Path) -> None:
    session = _bare_worker_session()
    manifest = {
        "model": {"sha256": "new-sha"},
        "trainer_step": 449,
        "recommended_worker": {"sf_nodes": 200},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    swaps: list[str] = []

    def _swap_model_from_manifest(manifest: dict) -> None:
        swaps.append(str(manifest["model"]["sha256"]))
        session.model_sha = str(manifest["model"]["sha256"])

    session._resolve_local_manifest_path = lambda: manifest_path
    cast(Any, session)._swap_model_from_manifest = _swap_model_from_manifest

    WorkerSession._check_model_update(session)

    assert session._stop_selfplay is True
    assert swaps == ["new-sha"]
    assert session.model_sha == "new-sha"


def test_periodic_manifest_poll_swaps_before_reco_restart() -> None:
    session = _bare_worker_session()
    manifest = {
        "task": {"type": "selfplay"},
        "model": {"sha256": "new-sha"},
        "trainer_step": 449,
        "recommended_worker": {"sf_nodes": 200},
    }

    swaps: list[str] = []

    def _swap_model_from_manifest(manifest: dict) -> None:
        swaps.append(str(manifest["model"]["sha256"]))
        session.model_sha = str(manifest["model"]["sha256"])

    session._poll_manifest = lambda: manifest
    cast(Any, session)._swap_model_from_manifest = _swap_model_from_manifest

    WorkerSession._periodic_manifest_poll(session)

    assert session._stop_selfplay is True
    assert swaps == ["new-sha"]
    assert session.model_sha == "new-sha"


def test_cp_wdl_recommendation_changes_restart_selfplay_session() -> None:
    session = _bare_worker_session()
    old = {
        "sf_nodes": 100,
        "sf_wdl_use_cp_logistic": False,
        "sf_wdl_cp_slope": 0.010,
        "sf_wdl_cp_draw_width": 60.0,
    }
    session._active_reco = {k: old.get(k) for k in WorkerSession._RECO_RESTART_KEYS}

    changed = WorkerSession._reco_changed(
        session,
        {
            "recommended_worker": {
                **old,
                "sf_wdl_use_cp_logistic": True,
            }
        },
        source_tag="test",
    )

    assert changed is True
    assert session._stop_selfplay is True


def test_threaded_local_model_swap_keeps_buffer_metadata_atomic(monkeypatch, tmp_path: Path) -> None:
    session = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.worker_model_update")
    session.inference_client = None
    session.model_sha = "old-sha"
    session.model_step = 1
    session.last_model_sha = "old-sha"
    session.model_cfg_active = ModelConfig(kind="tiny")
    session.cache_dir = tmp_path
    session.pending_dir = tmp_path / "pending"
    session.pending_dir.mkdir()
    session.trial_api_prefix = "/v1/trials/trial_00000"
    session.leased_trial_id = "trial_00000"
    session.fixed_trial_id = ""
    session.args = SimpleNamespace(username="worker")
    session.upload_buf = _BufferedUpload(positions=1, games=1, samples=[])
    lock = threading.Lock()
    session._upload_buf_lock = lock

    new_sha = "new-sha"
    (tmp_path / f"model_{new_sha}.pt").write_bytes(b"checkpoint")
    new_model = torch.nn.Identity()
    evaluator = SimpleNamespace(model="old-model")
    cast(Any, session)._direct_evaluator = evaluator

    session._server_url_for = lambda endpoint: f"http://server{endpoint}"
    session._load_and_compile_model = lambda *_args, **_kwargs: new_model
    monkeypatch.setattr(worker_mod, "_sha256_file", lambda _path: new_sha)

    sync_events: list[tuple[bool, str, int]] = []
    upload_events: list[bool] = []

    def _fake_flush(**kwargs):
        assert lock.locked()
        buf = kwargs["buf"]
        assert buf.positions == 1
        buf.positions = 0
        buf.samples = []
        return tmp_path / "pending.zarr", 7.0

    def _fake_sync(evaluator_arg, model_arg) -> None:
        sync_events.append((lock.locked(), session.model_sha, session.upload_buf.positions))
        evaluator_arg.model = model_arg

    def _fake_upload_pending_shards(*, default_elapsed_s: float | None = None) -> float:
        upload_events.append(lock.locked())
        assert default_elapsed_s == 7.0
        return 123.0

    monkeypatch.setattr(worker_mod, "_flush_upload_buffer_to_pending", _fake_flush)
    monkeypatch.setattr(worker_mod, "_sync_evaluator_to_model", _fake_sync)
    cast(Any, session)._upload_pending_shards = _fake_upload_pending_shards

    WorkerSession._swap_model_from_manifest(
        session,
        {"model": {"sha256": new_sha}, "trainer_step": 2},
    )

    assert sync_events == [(True, "old-sha", 0)]
    assert upload_events == [False]
    assert evaluator.model is new_model
    assert session.model is new_model
    assert session.model_sha == new_sha
    assert session.model_step == 2
    assert session.last_model_sha == new_sha
