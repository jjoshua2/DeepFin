from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from chess_anti_engine.server.app import create_app
from chess_anti_engine.server.auth import ensure_user


def _write_manifest(root: Path, trial_id: str) -> None:
    publish = root / "trials" / trial_id / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    (publish / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "server_version": "0.0.1",
                "min_worker_version": "0.0.1",
                "trial_id": trial_id,
                "task": {"type": "selfplay"},
            }
        ),
        encoding="utf-8",
    )


def test_lease_trial_is_sticky_and_balanced(tmp_path: Path) -> None:
    _write_manifest(tmp_path, "trial_a")
    _write_manifest(tmp_path, "trial_b")
    ensure_user(tmp_path / "users.json", username="worker", password="pw")

    client = TestClient(create_app(server_root=tmp_path))

    r1 = client.post("/v1/lease_trial", json={"worker_info": {"hostname": "box1"}}, auth=("worker", "pw"))
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["trial_id"] == "trial_a"
    assert d1["api_prefix"] == "/v1/trials/trial_a"

    r2 = client.post("/v1/lease_trial", json={"worker_info": {"hostname": "box2"}}, auth=("worker", "pw"))
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["trial_id"] == "trial_b"

    r3 = client.post(
        "/v1/lease_trial",
        json={"lease_id": d1["lease_id"], "trial_id": d1["trial_id"], "worker_info": {"hostname": "box1"}},
        auth=("worker", "pw"),
    )
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3["lease_id"] == d1["lease_id"]
    assert d3["trial_id"] == d1["trial_id"]
