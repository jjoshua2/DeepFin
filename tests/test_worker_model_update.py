from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from chess_anti_engine.worker import WorkerSession


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

    def _swap_model_from_manifest(m: dict) -> None:
        swaps.append(str(m["model"]["sha256"]))
        session.model_sha = str(m["model"]["sha256"])

    session._resolve_local_manifest_path = lambda: manifest_path
    session._swap_model_from_manifest = _swap_model_from_manifest

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

    def _swap_model_from_manifest(m: dict) -> None:
        swaps.append(str(m["model"]["sha256"]))
        session.model_sha = str(m["model"]["sha256"])

    session._poll_manifest = lambda: manifest
    session._swap_model_from_manifest = _swap_model_from_manifest

    WorkerSession._periodic_manifest_poll(session)

    assert session._stop_selfplay is True
    assert swaps == ["new-sha"]
    assert session.model_sha == "new-sha"
